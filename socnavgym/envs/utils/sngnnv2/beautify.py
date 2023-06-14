import cv2

def toColour(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def beautify_grey_image(img):
    return beautify_image(toColour(img))


def beautify_image(img):
    colours = [[150., 0., 0.],
			   [0., 0., 0.],
			   [127., 127., 127.],
			   [255., 255., 255.]]

    def colour_index(v):
        return int(v*(len(colours)-1))

    def value_for_index(index):
        return float(index)/(len(colours)-1)

    def convert(value):
        v = float(value)/255.
        if v > 0.999999:
            v = 0.999999
        if v < 0.01:
            v = 0.
        a = colours[colour_index(v)]
        b = colours[colour_index(v)+1]
        distance_to_base = abs(v - value_for_index(colour_index(v)))
        b_factor = distance_to_base*(len(colours)-1)
        a_factor = 1. - b_factor
        if a_factor > 1. or a_factor < 0.:
            sys.exit(0)

        r = (a_factor*a[0] + b_factor*b[0])
        g = (a_factor*a[1] + b_factor*b[1])
        b = (a_factor*a[2] + b_factor*b[2])
        return r, g, b

    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            v = float(img[row, col, 2])
            red, blue, green = convert(v)
            img[row, col, 0] = blue
            img[row, col, 1] = green
            img[row, col, 2] = red
    return img
