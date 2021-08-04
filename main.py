import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(path='data/eu4_map.png'):
    img = mpimg.imread(path)
    print(img)
    plt.imshow(img)
    plt.show()


def are_colinear(vec_1, vec_2, vec_3):
    return np.matmul((vec_1-vec_2),(vec_3-vec_2)) == np.linalg.norm(vec_1-vec_2) * np.linalg.norm(vec_3-vec_2)


def indices_by_closest(vectors, vec):
    distances = vectors - vec
    distances = np.linalg.norm(distances, axis=1)
    u, ind = np.unique(distances, return_index=True)
    return ind

def compute_forces(grid_pos, activated_points):
    return

class Projector():
    def __init__(self, base_path='data/eu4_map.png', to_project_path='data/chinese_provinces.png'):
        # Basic data
        self.delete_key = 'delete'
        self.go_back_key = 'b'
        self.fit_key = 'c'
        self.delete = 0

        # Load images
        self.base = mpimg.imread(base_path)
        self.base_x = len(self.base)
        self.base_y = len(self.base[0])
        self.initial = mpimg.imread(to_project_path)
        self.initial_x = len(self.initial)
        self.initial_y = len(self.initial[0])

        self.projected = np.copy(self.initial)

        self.build_model()
        # Create axes
        self.fig = plt.figure(figsize=(12, 10))
        self.ax_base = self.fig.add_subplot(2, 2, 1)
        self.ax_initial = self.fig.add_subplot(2, 2, 2)
        self.ax_deformed = self.fig.add_subplot(2, 2, 3)
        self.ax_superposed = self.fig.add_subplot(2, 2, 4)

        """self.ax_base.axis('off')
        self.ax_initial.axis('off')
        self.ax_deformed.axis('off')
        self.ax_superposed.axis('off')"""

        # Plot original images
        self.ax_base.imshow(self.base)
        self.ax_initial.imshow(self.initial)

        self.plot_points_base = self.ax_base.plot([0],[0],'+r')[0]
        self.plot_points_initial = self.ax_initial.plot([0],[0],'r+')[0]

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.fixed_points_initial = [] # pair of corresponding points from image to project
        self.fixed_points_base = []    # pair of corresponding points from base image
        plt.show()

    def build_model(self):
        grid = np.array([[i, j] for j in range(self.initial_x) for i in range(self.initial_y)])
        normalizer = normalize(grid)
        self.model = build_model(normalizer, [100, 100, 100])

    def fit(self):
        n_examples = min(len(self.fixed_points_initial), len(self.fixed_points_base))

        grid_initial = np.array([[i, j] for j in range(0,self.initial_x, 10) for i in range(0,self.initial_y, 10)]) / np.array([self.initial_x, self.initial_y])

        self.model.fit(grid_initial, grid_initial, epochs=100)

        """train_features = np.array(self.fixed_points_initial[:n_examples])
        train_labels = np.array(self.fixed_points_base[:n_examples]) / np.array([self.base_x, self.base_y])
        
        self.model.fit(train_features, train_labels, epochs=100)
        prediction = self.model.predict(train_features)
        print(train_labels)
        print(prediction)"""

    def predict_grid_nn(self):
        grid = np.array([[i, j] for j in range(self.initial_x) for i in range(self.initial_y)])
        simulated = self.model.predict(grid) * np.array([self.initial_x, self.initial_y])
        self.ax_initial.plot(simulated[:, 0], simulated[:, 1], '.')
        self.fig.canvas.draw()

    def predict_point_initial_triangulation(self, point):
        known_points = np.array(self.fixed_points_initial)
        closest = indices_by_closest(known_points, point)
        i = 2
        while are_colinear(known_points[closest[0]], known_points[closest[1]], known_points[closest[i]]):
            i += 1
            if i == len(known_points):
                print("All the points are colinear, need more points")
                return

        vec = point - known_points[closest[0]]
        b_1 = known_points[closest[1]] - known_points[closest[0]]
        b_2 = known_points[closest[i]] - known_points[closest[0]]

        b_12 = np.linalg.norm(b_1)**2
        b_22 = np.linalg.norm(b_2)**2

        b_1_b_2 = np.matmul(b_1, b_2)

        vec_b1 = np.matmul(vec, b_1)
        vec_b2 = np.matmul(vec, b_2)

        denominator = b_12 * b_22 - b_1_b_2**2

        coord_1 = b_22 * vec_b1 - b_1_b_2 * vec_b2
        coord_2 = b_12 * vec_b2 - b_1_b_2 * vec_b1

        coord_1 /= denominator
        coord_2 /= denominator

        known_points = np.array(self.fixed_points_base)
        b_0 = known_points[closest[0]]
        b_1 = known_points[closest[1]] - known_points[closest[0]]
        b_2 = known_points[closest[i]] - known_points[closest[0]]

        return b_1 * coord_1 + b_2 * coord_2 + b_0

    def predict_grid_triangulation(self):
        if len(self.fixed_points_base) < 3 or len(self.fixed_points_initial) < 3:
            print('Need more points')
            return
        grid = np.array([[i, j] for j in range(self.initial_x) for i in range(self.initial_y)])
        simulated = np.array([self.predict_point_initial_triangulation(point) for point in grid])
        self.ax_base.plot(simulated[:, 0], simulated[:, 1], '.')
        self.fig.canvas.draw()

    def plot_grid(self):
        grid = np.array([[i, j] for j in range(self.initial_x) for i in range(self.initial_y)])
        self.ax_initial.plot(grid[:, 0], grid[:, 1], '.')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == self.delete_key:
            self.delete = 1 - self.delete
        elif event.key ==self.go_back_key:
            print('To implement')
        elif event.key == self.fit_key:
            self.fit()
            self.predict_grid_nn()
        elif event.key =='g':
            self.plot_grid()


    def on_click(self, event):
        if event.button != 3:
            return

        if event.inaxes == self.ax_initial:
            if self.delete:
                self.delete_fixed_points(event.xdata, event.ydata, self.fixed_points_initial)
            elif len(self.fixed_points_initial) > len(self.fixed_points_base):
                self.fixed_points_initial[-1][0] = event.xdata
                self.fixed_points_initial[-1][1] = event.ydata
            else:
                self.fixed_points_initial.append([event.xdata, event.ydata])

        elif event.inaxes == self.ax_base:
            if self.delete:
                self.delete_fixed_points(event.xdata, event.ydata, self.fixed_points_base)
            elif len(self.fixed_points_initial) < len(self.fixed_points_base):
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
        print(self.fixed_points_base)
        print(self.fixed_points_initial)
        self.plot_points_initial.set_data([point[0] for point in self.fixed_points_initial],
                                          [point[1] for point in self.fixed_points_initial])
        self.plot_points_base.set_data([point[0] for point in self.fixed_points_base],
                                       [point[1] for point in self.fixed_points_base])
        self.fig.canvas.draw()


def normalize(data):
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(np.array(data))
    return normalizer


def build_model(normalizer, layers, input_dim=2, output_dim=2, activation='relu', optimizer='adam',loss='mean_squared_error'):
    model = tf.keras.Sequential([normalizer])
    for i in range(len(layers)):
        neurons_no = layers[i]
        # The first layer needs to know the number of inputs
        if i==0:
            model.add(tf.keras.layers.Dense(neurons_no, input_dim=input_dim, activation=activation ))
        else:
            model.add(tf.keras.layers.Dense(neurons_no, activation=activation))
    model.add(tf.keras.layers.Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__ == '__main__':
    Projector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
