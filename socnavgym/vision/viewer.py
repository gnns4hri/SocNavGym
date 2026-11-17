"""The movable 3D view (an orbiting camera whose pictures the render server makes), the windows that show it and the sensors'
readings, and the menu on the simulator's 2D window that shows and hides those windows. Windows are pygame's (SocNavGym already
uses pygame for its 2D window); nothing here is imported unless the view or a window is asked for.

The view starts from straight above and frames the whole scene by itself (it widens when the robot or a person leaves the map)
until the user moves or zooms it.
Mouse, on the 3D view:  left drag: turn around the point looked at (the framing goes on);  right drag (or shift + left drag): move
that point over the floor;  wheel: closer / further.   Keys (in any window):  arrows turn,  + / - zoom,  F: follow the robot (toggle),
R: back to the start view (from above, framing everything).
"""
import math
import numpy as np
from .ipc import CLASSES


class OrbitCamera:
    """The free view: a camera at `distance` from `target`, seen from `azimuth` / `elevation` (rad).
    Default: from straight above (elevation 90 degrees) with the world's +X to the right and +Y up, as in the simulator's 2D window,
    and `auto` on: before every picture the view is framed on the whole scene (frame), so everything is in the picture and it
    widens when the robot or a person leaves the map. Turning the view keeps `auto`; moving it, zooming or following the robot
    switches it off (the view is then the user's); reset brings back the starting view.
    Configuration (all optional): azimuth, elevation (degrees), fit (true / false = auto), margin, follow, and distance / target,
    which fix the view (auto off unless `fit: true`)."""
    def __init__(self, cfg=None):
        cfg = cfg or {}; self.target = np.array(cfg.get("target", (0.0, 0.0, 0.0)), float); self.distance = float(cfg.get("distance", 12.0))
        self.azimuth = math.radians(float(cfg.get("azimuth", -90.0))); self.elevation = float(np.clip(math.radians(float(cfg.get("elevation", 90.0))), 0.03, math.pi / 2))
        self.follow = bool(cfg.get("follow", False)); self.margin = float(cfg.get("margin", 1.08))
        self.auto = bool(cfg.get("fit", not (self.follow or "distance" in cfg or "target" in cfg)))
        self.home = (self.target.copy(), self.distance, self.azimuth, self.elevation, self.auto, self.follow)

    def reset(self):
        self.target, self.distance, self.azimuth, self.elevation, self.auto, self.follow = self.home[0].copy(), *self.home[1:]

    def axes(self):
        """The camera's axes in the world: X forward (towards the target), Y left, Z up. Defined from straight above too."""
        ce, se = math.cos(self.elevation), math.sin(self.elevation); ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        x = -np.array([ce * ca, ce * sa, se]); y = np.array([sa, -ca, 0.0]); return x, y, np.cross(x, y)

    def position(self): return self.target - self.distance * self.axes()[0]

    def pose(self):
        """4x4 pose of the camera in the world: X forward (towards the target), Y left, Z up."""
        x, y, z = self.axes(); M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = self.position(); return M

    def frame(self, lo, hi, hfov, aspect):
        """Aims at the centre of the box lo..hi (world, 3 numbers each) from the present direction, as close as shows all of it:
        hfov = horizontal field of view (rad), aspect = width / height of the picture; `margin` leaves some room around."""
        lo = np.asarray(lo, float); hi = np.asarray(hi, float); c = (lo + hi) / 2; x, y, z = self.axes(); th = math.tan(hfov / 2); tv = th / aspect; d = 0.5
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                for cz in (lo[2], hi[2]):
                    r = np.array([cx, cy, cz]) - c; d = max(d, abs(r @ y) / th - r @ x, abs(r @ z) / tv - r @ x)
        self.target = c; self.distance = float(d * self.margin)

    def orbit(self, dx, dy):
        self.azimuth -= 0.006 * dx; self.elevation = float(np.clip(self.elevation + 0.006 * dy, 0.03, math.pi / 2))

    def pan(self, dx, dy):
        k = 0.0016 * self.distance; fwd = np.array([-math.cos(self.azimuth), -math.sin(self.azimuth)]); left = np.array([-fwd[1], fwd[0]])
        self.target[:2] += k * (dx * left + dy * fwd); self.follow = False; self.auto = False

    def zoom(self, steps):
        self.distance = float(np.clip(self.distance * 0.9 ** steps, 0.5, 2000.0)); self.auto = False


# ---------------------------------------------------------------------- pictures of the readings
def depth_picture(depth, dmax):
    """Depth in metres -> a colour picture (near: warm, far: cold, no reading: black)."""
    v = np.clip(depth / max(min(dmax, 10.0), 1e-6), 0, 1); r = np.clip(1.5 - 4 * np.abs(v - 0.25), 0, 1); g = np.clip(1.5 - 4 * np.abs(v - 0.5), 0, 1); b = np.clip(1.5 - 4 * np.abs(v - 0.75), 0, 1)
    im = (np.stack([r, g, b], -1) * 255).astype(np.uint8); im[depth <= 0] = 0; return im


PALETTE = np.array([(0, 0, 0), (120, 110, 100), (200, 200, 190), (160, 110, 60), (60, 90, 200), (50, 160, 60), (150, 150, 160), (230, 60, 60), (240, 160, 30), (235, 235, 230)] + [(255, 255, 255)] * 246, np.uint8)


def label_picture(seg):
    return PALETTE[seg]


def scan_picture(ranges, azimuths, rmax, seg=None, size=360):
    """A planar scan seen from above, the sensor at the centre looking up the picture."""
    im = np.full((size, size, 3), 24, np.uint8); c = size / 2; k = (size / 2 - 6) / min(rmax, 12.0); ok = np.isfinite(ranges) & (ranges > 0)
    x = c - k * ranges[ok] * np.sin(azimuths[ok]); y = c - k * ranges[ok] * np.cos(azimuths[ok]); col = PALETTE[seg[ok]] if seg is not None else np.full((int(ok.sum()), 3), 255, np.uint8)
    xi = np.clip(x.astype(int), 1, size - 2); yi = np.clip(y.astype(int), 1, size - 2)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1): im[yi + dy, xi + dx] = col
    im[int(c) - 2:int(c) + 3, int(c) - 2:int(c) + 3] = (240, 160, 30); return im


# ---------------------------------------------------------------------- windows
class Windows:
    """The windows of the rendered pictures (one per name, besides the simulator's own 2D window) and the menu on the 2D window
    that shows and hides them. A sensor's window has the sensor's resolution and cannot be resized; a `resizable` one (the 3D
    view) can, and its picture is then rendered at the window's size."""
    BAR = 26                                          # height of the menu bar at the top of the 2D window (pixels)

    def __init__(self):
        import pygame
        from pygame._sdl2 import video
        self.pg = pygame; self.video = video; self.wins = {}; self.alive = True; self.open = False; self.rects = {}; self.font = None
        self.held = set(); self.pressed = []          # keys held down now / pressed since the last call of events() (pygame key codes)
        if not pygame.display.get_init(): pygame.display.init()

    def show(self, name, image, title=None, resizable=False):
        pg = self.pg; h, w = image.shape[:2]
        if name not in self.wins:
            win = self.video.Window(title or name, size=(w, h), resizable=resizable); self.wins[name] = (win, self.video.Renderer(win), resizable)
        win, ren, rs = self.wins[name]
        if not rs and tuple(win.size) != (w, h): win.size = (w, h)
        surf = pg.image.frombuffer(np.ascontiguousarray(image).tobytes(), (w, h), "RGB")
        tex = self.video.Texture.from_surface(ren, surf); ren.clear(); tex.draw(); ren.present()

    def hide(self, name):
        if name in self.wins:
            win, ren, _ = self.wins.pop(name)
            try: del ren; win.destroy()               # the renderer first: destroying the window frees it, and it must not be freed twice
            except Exception: pass

    def size(self, name):
        """(width, height) of the window `name` now, or None when it is not open."""
        return (max(int(self.wins[name][0].size[0]), 64), max(int(self.wins[name][0].size[1]), 64)) if name in self.wins else None

    def menu(self, screen, shown):
        """Draws the menu on the 2D window `screen` (a pygame surface): a bar with "Windows" and, when it was clicked, the list of
        the windows in `shown` (name -> shown?) with a tick each."""
        pg = self.pg
        if self.font is None: pg.font.init(); self.font = pg.font.Font(None, 22)
        ink = (20, 20, 20); B = self.BAR; W = screen.get_width(); f = self.font; t = f.render("Windows", True, ink); r = pg.Rect(0, 0, t.get_width() + 34, B - 1); self.rects = {"": r}
        pg.draw.rect(screen, (236, 236, 236), (0, 0, W, B)); pg.draw.line(screen, (170, 170, 170), (0, B - 1), (W, B - 1))
        if self.open: pg.draw.rect(screen, (205, 215, 235), r)
        screen.blit(t, (10, (B - t.get_height()) // 2)); x = t.get_width() + 16; y = B // 2 - 2; pg.draw.polygon(screen, ink, [(x, y), (x + 8, y), (x + 4, y + 5)])
        if not self.open: return
        h = 24; w = max(f.size(n)[0] for n in shown) + 46; box = pg.Rect(0, B - 1, w, h * len(shown) + 6); pg.draw.rect(screen, (246, 246, 246), box); pg.draw.rect(screen, (170, 170, 170), box, 1)
        for i, (n, on) in enumerate(shown.items()):
            y = B + 2 + i * h; self.rects[n] = pg.Rect(0, y, w, h); c = pg.Rect(10, y + 5, 14, 14); pg.draw.rect(screen, (255, 255, 255), c); pg.draw.rect(screen, (90, 90, 90), c, 1)
            if on: pg.draw.lines(screen, (20, 120, 40), False, [(c.x + 3, c.y + 7), (c.x + 6, c.y + 10), (c.x + 11, c.y + 3)], 2)
            t = f.render(n, True, ink); screen.blit(t, (34, y + (h - t.get_height()) // 2))

    def click(self, pos, shown):
        if self.rects.get("") is not None and self.rects[""].collidepoint(pos): self.open = not self.open; return
        if self.open:
            for n, r in self.rects.items():
                if n and r.collidepoint(pos): shown[n] = not shown[n]; return          # the list stays open: several windows can be switched in a row
        self.open = False

    def events(self, cam=None, shown=None, view=None):
        """Handles the pending events of all the windows: the menu on the 2D window (`shown`: name -> shown?, changed in place), a
        rendered window's close button (hides it), the mouse on the 3D view's window (named `view`) and the keys.
        False once the 2D window was closed or Q / Esc pressed."""
        pg = self.pg; self.pressed = []
        # An event of a rendered window carries that window's object; one of the 2D window carries none (or the object a program
        # made for it). The 2D window is NOT asked for with Window.from_display_module() here: pygame stores a bare pointer to the
        # object that call returns in the 2D window and hands it out with every later event of that window, also once the object
        # is gone (a crash at some later moment).
        ours = {w[0].id for w in self.wins.values()}
        for e in pg.event.get():
            wid = getattr(getattr(e, "window", None), "id", None); vid = self.wins[view][0].id if view in self.wins else None
            if e.type == pg.KEYDOWN: self.held.add(e.key); self.pressed.append(e.key)
            elif e.type == pg.KEYUP: self.held.discard(e.key)
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)): self.alive = False
            elif e.type == getattr(pg, "WINDOWCLOSE", -1):
                name = next((n for n, w in self.wins.items() if w[0].id == wid), None)
                if name is None: self.alive = False                         # the 2D window
                else:
                    self.hide(name)
                    if shown is not None and name in shown: shown[name] = False
            elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1 and shown and wid not in ours: self.click(e.pos, shown)
            if cam is None: continue
            if e.type == pg.MOUSEMOTION and (e.buttons[0] or e.buttons[2]) and wid == vid:
                if e.buttons[2] or (pg.key.get_mods() & pg.KMOD_SHIFT): cam.pan(*e.rel)
                else: cam.orbit(*e.rel)
            elif e.type == pg.MOUSEWHEEL and wid == vid: cam.zoom(e.y)
            elif e.type == pg.KEYDOWN:                                      # from any of the windows
                if e.key == pg.K_LEFT: cam.orbit(40, 0)
                elif e.key == pg.K_RIGHT: cam.orbit(-40, 0)
                elif e.key == pg.K_UP: cam.orbit(0, 25)
                elif e.key == pg.K_DOWN: cam.orbit(0, -25)
                elif e.key in (pg.K_PLUS, pg.K_KP_PLUS, pg.K_EQUALS): cam.zoom(1)
                elif e.key in (pg.K_MINUS, pg.K_KP_MINUS): cam.zoom(-1)
                elif e.key == pg.K_f: cam.follow = not cam.follow; cam.auto = False
                elif e.key == pg.K_r: cam.reset()
        return self.alive

    def close(self):
        for n in list(self.wins): self.hide(n)
