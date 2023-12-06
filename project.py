##Libraries
import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')
from scipy.signal import convolve2d
import yaml

##Functions
def apply_filter_to_patch(patch, filter):
    result = cv2.filter2D(patch, -1, filter)
    return result

def apply_filter_to_image(image, filter):
    h, w = filter.shape
    half_h, half_w = h // 2, w // 2

    image = cv2.copyMakeBorder(image, half_h, half_h, half_w, half_w, cv2.BORDER_REFLECT)

    output = np.zeros_like(image)

    for i in range(half_h, image.shape[0] - half_h):
        for j in range(half_w, image.shape[1] - half_w):
            patch = image[i - half_h:i + half_h + 1, j - half_w:j + half_w + 1]
            filtered_patch = apply_filter_to_patch(patch, filter)
            output[i, j] = filtered_patch[half_h, half_w]
            
    output = output[half_h:-half_h, half_w:-half_w]

    return output

def npapply_filter_to_image(image, filter):
    h, w = filter.shape
    half_h, half_w = h // 2, w // 2

    image = cv2.copyMakeBorder(image, half_h, half_h, half_w, half_w, cv2.BORDER_REFLECT, value=0)

    output = np.zeros_like(image)

    for channel in range(image.shape[2]):
        filtered_channel = convolve2d(image[:, :, channel], filter, mode='same')
        output[:, :, channel] = filtered_channel

    output = output[half_h:-half_h, half_w:-half_w, :]

    return output

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist

def draw_hist(canvas, figure):
   tkcanvas = FigureCanvasTkAgg(figure, canvas)
   tkcanvas.draw()
   tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
   
def update_hist(fig, hist1, hist2):
    ax = fig.get_axes()[0]
    ax.clear()
    ax.bar(np.arange(len(hist1)), hist1, color='blue', alpha=0.5, label='Original Histogram')
    ax.bar(np.arange(len(hist2)), hist2, color='red', alpha=0.5, label='Updated Histogram')
    plt.title('Histogram')
    fig.canvas.draw()
    
def greyScale(np_image):
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    return image

def histogramEqualization(np_image):
    if len(np_image.shape) == 2:  # Grayscale image
        sg.popup_error('Warning: Histogram equalization is not applicable to grayscale images.')
        return np_image
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    v = cv2.equalizeHist(v)
    image = cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def averagingFilter(np_image, filter_size):
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)
    filtered_image = npapply_filter_to_image(np_image, kernel)
    return filtered_image

def gaussianFilter(np_image, kernel_size):
    sigma = kernel_size / 3
    gaussian_filter = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_filter = np.outer(gaussian_filter, gaussian_filter)
    gaussian_filter /= np.sum(gaussian_filter)  
    filtered_image = npapply_filter_to_image(np_image, gaussian_filter)
    return filtered_image

def NNResize(newWidth, tempWidth, newHeight, tempHeight, np_image, channels):
    x_scale = newWidth / tempWidth
    y_scale = newHeight / tempHeight
    image = np.zeros((newHeight, newWidth, channels), dtype=np_image.dtype)
    for y in range(newHeight):
        for x in range(newWidth):
            src_x = int(x / x_scale)
            src_y = int(y / y_scale)
            src_x = np.clip(src_x, 0, tempWidth - 1)
            src_y = np.clip(src_y, 0, tempHeight - 1)
            image[y, x, :] = np_image[src_y, src_x, :]
    return image

def BilinearResize(newWidth, tempWidth, newHeight, tempHeight, np_image, channels):
    x_scale = newWidth / tempWidth
    y_scale = newHeight / tempHeight

    image = np.zeros((newHeight, newWidth, channels), dtype=np_image.dtype)

    for y in range(newHeight):
        for x in range(newWidth):
            src_x = int(x / x_scale)
            src_y = int(y / y_scale)
            src_x = np.clip(src_x, 0, tempWidth - 1)
            src_y = np.clip(src_y, 0, tempHeight - 1)
            x1 = int(src_x)
            x2 = min(x1 + 1, tempWidth - 1)
            y1 = int(src_y)
            y2 = min(y1 + 1, tempHeight - 1)

            alpha = src_x - x1
            beta = src_y - y1

            for c in range(channels):
                image[y, x, c] = (1 - alpha) * (1 - beta) * np_image[y1, x1, c] + alpha * (1 - beta) * np_image[y1, x2, c] + \
                                (1 - alpha) * beta * np_image[y2, x1, c] + alpha * beta * np_image[y2, x2, c]
    return image
    
def save_settings(saturation_change, contrast_change, color_palette_change):
    settings = {
        'saturation_change': saturation_change,
        'contrast_change': contrast_change,
        'color_palette_change': color_palette_change
    }
    
    save_file = sg.popup_get_file('Save Settings As', save_as=True, file_types=(("YAML files", "*.yaml"),))

    if save_file:
        with open(save_file, 'w') as file:
            yaml.dump(settings, file)
            
def load_settings():
    load_file = sg.popup_get_file('Load Settings', file_types=(("YAML files", "*.yaml"),))

    if load_file:
        with open(load_file, 'r') as file:
            settings = yaml.load(file, Loader=yaml.FullLoader)

        if settings:
            return settings.get('saturation_change', 0), settings.get('contrast_change', 0), settings.get('color_palette_change', 0)

    return 0, 0, 0

def add_scratches_and_dust(original_image, overlay_image_path, opacity, angle):
    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
    overlay_image = cv2.resize(overlay_image, (original_image.shape[1], original_image.shape[0]))

    rotation_matrix = cv2.getRotationMatrix2D((overlay_image.shape[1] // 2, overlay_image.shape[0] // 2), angle, 1)
    overlay_image = cv2.warpAffine(overlay_image, rotation_matrix, (overlay_image.shape[1], overlay_image.shape[0]))

    alpha_channel = overlay_image[:, :, 3] / 255.0
    alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])

    original_image = original_image.astype(float)
    overlay_image = overlay_image[:, :, :3].astype(float)

    blended_image = original_image * (1 - opacity) + overlay_image[:, :, :3] * opacity
    result = blended_image * alpha_mask + original_image * (1 - alpha_mask)

    return result.astype(np.uint8)

def resetLayout1(width, height):
    layout = [
        [
            sg.Graph(
                canvas_size=(width*2, height),
                graph_bottom_left=(0, 0),
                graph_top_right=(width*2, height),
                key='-IMAGE-',
                background_color='white',
                change_submits=True,
                drag_submits=True
            ),
            sg.Column([
                [sg.Canvas(key='-HIST-', size=(600, height))],
            ]),
            [sg.Button('Original', size=(10, 2)),sg.Button('Exit', size=(10, 2))],
            [sg.Button('Greyscale', size=(10, 2)), 
             sg.Button('Histogram Equalization', size=(10, 2)), 
             sg.Button('Filter', size=(10, 2)),
             sg.Button('Resize', size=(10, 2)), 
             sg.Button('Paint', size=(10, 2)),
             sg.Button('Scratch and Dust', key='-scratch-', size=(10, 2)),
             sg.Button('Save Image', key='-save-', size=(10, 2)),
             sg.Button('Load Image', key='-load-', size=(10, 2))
            ],
            [sg.Slider(range=(0, 15), default_value=0, orientation='h', size=(10, 30), key='-SLIDER1-'), sg.Button('Apply averaging Filter')],
            [sg.Slider(range=(0, 15), default_value=0, orientation='h', size=(10, 30), key='-SLIDER2-'), sg.Button('Apply Gaussian Filter')],

        ]
    ]
    return layout

def resetLayout2():
    layout2 = [
        [sg.Text("height: "), sg.InputText('', key='-height-'), sg.Text('px')],
        [sg.Text("width: "), sg.InputText('', key='-width-'), sg.Text('px')],
        [sg.Checkbox('Constrained', key='-Constrained-')],
        [sg.Button("N|N"), sg.Button("PopUpExit")],
        [sg.Button("Bilinear")]
    ]
    return layout2

def resetLayout3(width, height):
    layout3 = [
        [sg.Graph(
            canvas_size=(width*2, height),
            graph_bottom_left=(0, 0),
            graph_top_right=(width*2, height),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            drag_submits=True
        )],
        [sg.Text('Saturation: ', size=(10,1)), sg.Slider(range=(-10, 10), default_value=0, orientation='h', size=(10,30), key='-Saturation-'), sg.Button('Save Settings'), sg.Button('Load Settings')],
        [sg.Text('Contrast: ', size=(10,1)), sg.Slider(range=(-10, 10), default_value=0, orientation='h', size=(10,30), key='-Contrast-')],
        [sg.Text('ColorPallete: ', size=(10,1)), sg.Slider(range=(-10, 10), default_value=0, orientation='h', size=(10,30), key='-ColorPalette-'), sg.Button('Apply'), sg.Button('Quit')]
    ]
    return layout3

def resetLayout4(width, height):
    layout4 = [
        [sg.Graph(
            canvas_size=(width*2, height),
            graph_bottom_left=(0, 0),
            graph_top_right=(width*2, height),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            drag_submits=True
        )],
        [sg.Button('Original', key='original'), 
         sg.Button('Filter 1', key='Filter1'), 
         sg.Button('Filter 2', key='Filter2'), 
         sg.Button('Filter 3', key='Filter3'), 
         sg.Button('Filter 4', key='Filter4'),
         sg.Button('Filter 5', key='Filter5'), 
         sg.Button('Quit')
         ],
        [sg.Text('Opacity:', size=(10, 1)), sg.Slider(range=(0,100), default_value=0, orientation='h',size=(50,30), key='-Opacity-')],
        [sg.Text('Angle:', size=(10, 1)), sg.Slider(range=(0,360), default_value=0, orientation='h',size=(50,30), key='-Angle-')]
    ]
    return layout4

def compute_gradients(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude, gradient_direction = cv2.cartToPolar(gradient_x, gradient_y)
    return gradient_magnitude, gradient_direction

def clip_point(x, y, width, height):
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y

def paint(image, num_pixels=10000, stroke_length_range=(25, 30), stroke_width_range=(25, 30), gradient_threshold=50):
    height, width, _ = image.shape

    gradient_magnitude, gradient_direction = compute_gradients(image)
    pixels_to_paint = np.where(gradient_magnitude > gradient_threshold)
    indices = np.random.choice(len(pixels_to_paint[0]), min(num_pixels, len(pixels_to_paint[0])), replace=False)
    pixels_to_paint = (pixels_to_paint[0][indices], pixels_to_paint[1][indices])

    for i in range(len(pixels_to_paint[0])):
        y, x = pixels_to_paint[0][i], pixels_to_paint[1][i]

        stroke_width = np.random.randint(stroke_width_range[0], stroke_width_range[1] + 1)
        angle = gradient_direction[y, x]
        stroke_length = np.random.randint(stroke_length_range[0], stroke_length_range[1] + 1)

        dx = int(stroke_length * np.cos(angle))
        dy = int(stroke_length * np.sin(angle))

        start_x, start_y = clip_point(x - dx // 2, y - dy // 2, width, height)
        end_x, end_y = clip_point(x + dx // 2, y + dy // 2, width, height)

        color = tuple(map(int, image[y, x]))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, stroke_width)
    return image


def display_image(width, height, np_image, beforeImage, originalImage):
    image_data = np_im_to_data(np_image)
    beforeimage_data = np_im_to_data(beforeImage)
    
    hist1 = construct_image_histogram(np_image)
    fig = plt.figure(figsize=(5,4),dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(hist1)), hist1, color='blue', alpha=0.5, label='Original Histogram')
    plt.title('Histogram')
    
    layout = resetLayout1(width, height)

    window = sg.Window('Display Image', layout, finalize=True)  
    window['-IMAGE-'].draw_image(data=beforeimage_data, location=(0, height))  
    window['-IMAGE-'].draw_image(data=image_data, location=(width, height))
    draw_hist(window['-HIST-'].TKCanvas, fig)
    
    layout2 = resetLayout2()

    windowPopUp = sg.Window("Resize", layout2)
    
    layout3 = resetLayout3(width, height)
    layout4 = resetLayout4(width, height)
    
    
    while True:
        event, values = window.read()
        if event == 'Greyscale':
            np_image = greyScale(np_image)
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))
            
        if event == 'Histogram Equalization':
            np_image = histogramEqualization(np_image)
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))
            hist2 = construct_image_histogram(np_image)
            update_hist(fig, hist1, hist2)

        if event == 'Original':
            np_image = originalImage.copy()
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))
            
        if event == 'Apply averaging Filter':
            filter_size = int(values['-SLIDER1-']) * 2 + 3
            np_image = averagingFilter(np_image, filter_size)
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))
            
        if event == 'Apply Gaussian Filter':
            kernel_size = int(values['-SLIDER2-']) * 2 + 1
            np_image = gaussianFilter(np_image, kernel_size)
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))

        if event == 'Resize':
            event, values = windowPopUp.read()
            
            if event == "N|N":
                tempHeight, tempWidth, channels = np_image.shape
                newWidth = int(values['-width-'])
                if values['-Constrained-']:
                    aspect_ratio = tempWidth / tempHeight
                    newHeight = int(newWidth / aspect_ratio)
                else:
                    newHeight = int(values['-height-'])
                if newHeight > 0 and newWidth > 0:
                    np_image = NNResize(newWidth, tempWidth, newHeight, tempHeight, np_image, channels)
                    window.close()
                    display_image(width, height, np_image, beforeImage, originalImage)
                
            if event == "Bilinear":
                tempheight, tempwidth, channels = np_image.shape
                newWidth = int(values['-width-'])
                if values['-Constrained-']:
                    aspect_ratio = tempwidth / tempheight
                    newHeight = int(newWidth / aspect_ratio)
                else:
                    newHeight = int(values['-height-'])
                if newHeight > 0 and newWidth > 0:
                    np_image = BilinearResize(newWidth, tempWidth, newHeight, tempHeight, np_image, channels)
                    window.close()
                    display_image(width, height, np_image, beforeImage, originalImage)
                
            if event == sg.WINDOW_CLOSED or event == 'PopUpExit':
                window.close()
                display_image(width, height, np_image, beforeImage, originalImage)
        if event == '-save-':
                save_file = sg.popup_get_file('Save As', save_as=True, file_types=(("PNG files", "*.png"),))
                if save_file:
                    array = np.array(image, dtype=np.uint8)
                    resized_image = Image.fromarray(array)
                    resized_image.save(save_file)
                    sg.popup(f"Image saved to {save_file}")
        if event == '-load-':
            file_path = sg.popup_get_file('Select Image File', file_types=(("Image files", "*.png;*.jpg;*.jpeg"),))
            if file_path:
                image = cv2.imread(file_path)
                if image is not None:                        
                    beforeImage = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    width = 640
                    scale = width/image.shape[1]
                    height = int(image.shape[0]*scale)
                    beforeImage = cv2.resize(beforeImage, (width,height), interpolation=cv2.INTER_LINEAR)
                    image = cv2.resize(image, (width,height), interpolation=cv2.INTER_LINEAR)
                    originalImage = image.copy()
                    window.close()
                    display_image(width, height, image, beforeImage, originalImage)
                else:
                    sg.popup_error(f"Error loading image from {file_path}")
        
        if event == 'Filter':
            initialImage = np_image
            initialImageData = np_im_to_data(initialImage)
            window.close()
            FilterPopUp = sg.Window('Filter', layout3, finalize=True)
            FilterPopUp['-IMAGE-'].draw_image(data=initialImageData, location=(0, height))  
            while True:
                event, values = FilterPopUp.read()
                saturationChange = int(values['-Saturation-'])
                contrastChange = int(values['-Contrast-'])
                ColorPaletteChange = int(values['-ColorPalette-'])
                if event == 'Apply':
                    image = initialImage
                    if saturationChange > 0 or saturationChange < 0:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        h, s, v = cv2.split(image)
                        saturationChange = saturationChange * 2
                        s = np.clip(s + saturationChange, 0, 255)
                        h = h.astype(np.uint8)
                        s = s.astype(np.uint8)
                        v = v.astype(np.uint8)
                        image = cv2.merge([h,s,v])
                        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                    if contrastChange > 0 or contrastChange < 0:
                        contrast_factor = 1.0 + contrastChange / 10.0  # You can adjust the factor for more or less contrast
                        lut = np.array([((i / 255.0) ** contrast_factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                        image = cv2.LUT(image, lut)
                    if ColorPaletteChange > 0 or ColorPaletteChange < 0:
                        image[:, :, 0] = np.clip(image[:, :, 0] + ColorPaletteChange, 0, 255)  # Adjusting blue channel
                        image[:, :, 2] = np.clip(image[:, :, 2] - ColorPaletteChange, 0, 255)
                        
                    filtered_image_data = np_im_to_data(image)
                    FilterPopUp['-IMAGE-'].draw_image(data=filtered_image_data, location=(width, height))
                    
                if event == 'Save Settings':
                    save_settings(saturationChange, contrastChange, ColorPaletteChange)
                
                if event == 'Load Settings':
                    saturationChange, contrastChange, ColorPaletteChange = load_settings()
                    FilterPopUp['-Saturation-'].update(saturationChange)
                    FilterPopUp['-Contrast-'].update(contrastChange)
                    FilterPopUp['-ColorPalette-'].update(ColorPaletteChange)
                            
                if event == 'Quit':
                    np_image = image.copy()
                    FilterPopUp.close()
                    display_image(width, height, np_image, beforeImage, originalImage)
        
        if event == 'Paint':
            np_image = paint(np_image, num_pixels=5000, stroke_length_range=(5, 10), stroke_width_range=(1, 2), gradient_threshold=50)
            image_data = np_im_to_data(np_image)
            window['-IMAGE-'].draw_image(data=image_data, location=(width, height))

        
        if event == '-scratch-':
            window.close()
            initialImage = np_image.copy()
            initialImageData = np_im_to_data(initialImage)
            FilterPopUp = sg.Window('Filter', layout4, finalize=True)
            FilterPopUp['-IMAGE-'].draw_image(data=initialImageData, location=(0, height))
            
            paths =['dust&scratches/overlay1.png', 'dust&scratches/overlay2.png', 'dust&scratches/overlay3.png', 'dust&scratches/overlay4.png', 'dust&scratches/overlay5.png']
            while True:
                event, values = FilterPopUp.read()
                opacity = int(values['-Opacity-'])
                angle = int(values['-Angle-'])
                if event == 'original':
                    np_image = initialImage.copy()
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Filter1':
                    np_image = add_scratches_and_dust(np_image, paths[0], opacity/100, angle)
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Filter2':
                    np_image = add_scratches_and_dust(np_image, paths[1], opacity/100, angle)
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Filter3':
                    np_image = add_scratches_and_dust(np_image, paths[2], opacity/100, angle)
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Filter4':
                    np_image = add_scratches_and_dust(np_image, paths[3], opacity/100, angle)
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Filter5':
                    np_image = add_scratches_and_dust(np_image, paths[4], opacity/100, angle)
                    imageData = np_im_to_data(np_image)
                    FilterPopUp['-IMAGE-'].draw_image(data=imageData, location=(width, height))
                if event == 'Quit':
                    FilterPopUp.close()
                    display_image(width, height, np_image, beforeImage, originalImage)
                       
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

    window.close() 


##Main
def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')
    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    beforeImage = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f'{image.shape}')

    print(f'Resizing the image to 480x640 ...', end='')
    width = 640
    scale = width/image.shape[1]
    height = int(image.shape[0]*scale)
    beforeImage = cv2.resize(beforeImage, (width,height), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, (width,height), interpolation=cv2.INTER_LINEAR)
    originalImage = image.copy()
    print(f'{image.shape}')

    display_image(width, height, image, beforeImage, originalImage)

if __name__ == '__main__':
    main()