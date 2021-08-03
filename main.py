import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(path='data/eu4_map.png'):
    img = mpimg.imread(path)
    print(img)
    plt.imshow(img)
    plt.show()


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

        self.plot_points_base = self.ax_base.plot([0],[0],'+r')[0]
        self.plot_points_initial = self.ax_initial.plot([0],[0],'r+')[0]

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.fixed_points_initial = [] # pair of corresponding points from image to project
        self.fixed_points_base = [] # pair of corresponding points from base image
        plt.show()


    def on_key_press(self, event):
        if event.key == self.delete_key:
            self.delete = 1 - self.delete
        elif event.key ==self.go_back_key:
            print('To implement')


    def on_click(self, event):
        if event.button != 3:
            return

        if event.inaxes == self.ax_initial:
            if self.delete:
                self.delete_fixed_points(event.xdata, event.ydata, self.fixed_points_initial)
            elif len(self.fixed_points_initial)>len(self.fixed_points_base):
                self.fixed_points_initial[-1][0] = event.xdata
                self.fixed_points_initial[-1][1] = event.ydata
            else:
                self.fixed_points_initial.append([event.xdata, event.ydata])

        elif event.inaxes == self.ax_base:
            if self.delete:
                self.delete_fixed_points(event.xdata, event.ydata, self.fixed_points_base)
            elif len(self.fixed_points_initial)<len(self.fixed_points_base):
                self.fixed_points_base[-1][0] = event.xdata
                self.fixed_points_base[-1][1] = event.ydata
            else:
                self.fixed_points_base.append([event.xdata, event.ydata])
        else:
            return

        self.update_fixed_points_plot()


    def delete_fixed_points(self, x, y, fixed_points):
        if fixed_points == []:
            return
        fixed_points = np.array(fixed_points)
        distances = fixed_points - np.array([x, y])
        distances = np.linalg.norm(distances, axis=1)
        closest_index = np.argmin(distances)
        if len(self.fixed_points_base)>closest_index:
            self.fixed_points_base.pop(closest_index)
        if len(self.fixed_points_initial)>closest_index:
            self.fixed_points_initial.pop(closest_index)


    def update_fixed_points_plot(self):
        self.plot_points_initial.set_data([point[0] for point in self.fixed_points_initial],
                                          [point[1] for point in self.fixed_points_initial])
        self.plot_points_base.set_data([point[0] for point in self.fixed_points_base],
                                       [point[1] for point in self.fixed_points_base])
        self.fig.canvas.draw()


if __name__ == '__main__':
    Projector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
