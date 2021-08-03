import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(path='data/eu4_map.png'):
    img = mpimg.imread(path)
    print(img)
    plt.imshow(img)
    plt.show()

def remove_last_none(array):
    if None in array[-1]:
        return array[:-1]


class Projector():
    def __init__(self, base_path='data/eu4_map.png', to_project_path='data/chinese_provinces.png'):
        # Basic data
        self.delete_key = 'delete'
        self.go_back_key = 'b'
        self.delete = 0

        # Load images
        self.base = mpimg.imread(base_path)
        self.initial = mpimg.imread(to_project_path)

        self.projected = np.copy(self.initial)

        # Create axes
        self.fig = plt.figure(figsize=(12, 10))
        self.ax_base = self.fig.add_subplot(2, 2, 1)
        self.ax_initial = self.fig.add_subplot(2, 2, 2)
        self.ax_deformed = self.fig.add_subplot(2, 2, 3)
        self.ax_superposed = self.fig.add_subplot(2, 2, 4)

        self.ax_base.axis('off')
        self.ax_initial.axis('off')
        self.ax_deformed.axis('off')
        self.ax_superposed.axis('off')

        # Plot original images
        self.ax_base.imshow(self.base)
        self.ax_initial.imshow(self.initial)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.fixed_points_initial = [] # pair of corresponding points from image to project
        self.fixed_points_base = [] # pair of corresponding points from base image
        plt.show()


    def on_key_press(self, event):
        if event.key == self.delete_key:
            self.delete_key = 1 - self.delete_key
        elif event.key ==self.go_back_key:
            print('To implement')

    def on_click(self, event):
        if event.button != 3:
            return
        if event.inaxes == self.ax_initial:
            self.fixed_points_initial = \
                self.modify_fixed_points(event.xdata, event.ydata, self.fixed_points_initial)
        elif event.inaxes == self.ax_base:
            self.fixed_points_base = \
                self.modify_fixed_points(event.xdata, event.ydata, self.fixed_points_base)
        else:
            return

        self.update_fixed_points_plot(self)


    def modify_fixed_points(self, x, y, fixed_points):
        if self.delete and self.fixed_points == []:
            return
        if self.delete:
            distances = fixed_points - np.array([x, y])
            distances = np.linalg.norm(distances, axis=1)
            closest_index = np.argmin(distances)
    def modify_base_fixed_points(self, x, y):



    def update_fixed_points_plot(self):
        self.ax_initial.plot(self.fixed_points_initial[:0], self.fixed_points_initial[:1], '.')
        self.ax_base.plot(self.fixed_points_base[:0], self.fixed_points_base[:1], '.')
        self.fig.draw()


if __name__ == '__main__':
    Projector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
