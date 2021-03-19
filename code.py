import math
from skimage import io, color
import numpy as np
from tqdm import trange
import scipy.ndimage as nd
import skimage.segmentation as sm
import cv2 as cv
import scipy.spatial.kdtree as kd
import skimage.filters as filters
import logging as log


class Cluster(object):
    cluster_index = 1

    def __init__(self, image_h, image_w, h, w, S, l_channel=0, a_channel=0, b_channel=0, ):
        self.S = S
        self.h = h
        self.w = w
        self.image_h = image_h
        self.image_w = image_w
        self.offset = int(self.S)
        self.h_low = self.h - self.offset
        self.w_low = self.w - self.offset
        self.h_high = self.h + self.offset
        self.w_high = self.w + self.offset
        self.h_low = self.h_low if self.h_low >= 0 else 0
        self.w_low = self.w_low if self.w_low >= 0 else 0
        self.h_high = self.h_high if self.h_high <= self.image_h - 1 else self.image_h - 1
        self.w_high = self.w_high if self.w_high <= self.image_w - 1 else self.image_w - 1

        self.l = l_channel
        self.a = a_channel
        self.b = b_channel
        self.index = self.cluster_index
        self.pixels = []
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.h_low = self.h - self.offset
        self.w_low = self.w - self.offset
        self.h_high = self.h + self.offset
        self.w_high = self.w + self.offset

        self.h_low = self.h_low if self.h_low >= 0 else 0
        self.w_low = self.w_low if self.w_low >= 0 else 0
        self.h_high = self.h_high if self.h_high <= self.image_h - 1 else self.image_h - 1
        self.w_high = self.w_high if self.w_high <= self.image_w - 1 else self.image_w - 1

        self.l = l
        self.a = a
        self.b = b

    def __eq__(self, o: object) -> bool:
        np.vectorize()
        if isinstance(o, Cluster):
            return self.index == o.index
        else:
            return False

    def __hash__(self):
        return self.index

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)


class SLICProcessor(object):

    def __init__(self, input_image, alpha, filename, K, std, rgb_color, post_process, max_iteration=20):
        self.K = K
        self.color = rgb_color
        self.alpha = alpha

        self.filename = filename
        self.data = input_image

        self.post_process = post_process
        self.image_height, self.image_width, _ = self.data.shape
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
        self.max_iteration = max_iteration

        self.copy = self.data.copy()

        self.std = std
        self.data = filters.gaussian(self.data, std, multichannel=False)

        self.data = color.rgb2lab(self.data)

        self.x_coordinate, self.y_coordinate = self.compute_xy_coordinate()
        self.gradiant_magnitude = self.compute_gradient_magnitude()

        self.clusters = set()
        self.kd_tree = None
        self.x_sequence = None
        self.y_sequence = None

        self.distance = np.full((self.image_height, self.image_width), np.inf)
        self.pixel_to_cluster = np.full((self.image_height, self.image_width), fill_value=0, dtype='int')

    def make_cluster(self, h, w):
        return Cluster(self.image_height, self.image_width,
                       h, w,
                       self.S,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def compute_xy_coordinate(self):
        heights_array = np.arange(0, self.image_height, 1)
        widths_array = np.arange(0, self.image_width, 1)
        widths, heights = np.meshgrid(widths_array, heights_array)
        return widths, heights

    def initialise_clusters(self):
        step = self.S
        self.x_sequence = np.arange(step, self.image_width, step).astype('int')
        self.y_sequence = np.arange(step, self.image_height, step).astype('int')

        self.clusters = {self.make_cluster(y, x) for x in self.x_sequence for y in self.y_sequence}

    def initial_move_clusters(self):
        delta = 5
        for cluster in self.clusters:
            cluster: Cluster
            w_low, w_high = cluster.w - delta, cluster.w + delta
            h_low, h_high = cluster.h - delta, cluster.h + delta
            part = self.gradiant_magnitude[h_low:h_high, w_low:w_high]
            # get argmin of area based on gradiant magnitude value
            col, row = SLICProcessor.min_index(part)
            new_w, new_h = w_low + col, h_low + row

            # set pixel's cluster
            self.pixel_to_cluster[new_h - 5:new_h + 5, new_w - 5:new_w + 5] = cluster.index
            cluster.update(new_h,
                           new_w,
                           self.data[new_h][new_w][0],
                           self.data[new_h][new_w][1],
                           self.data[new_h][new_w][2])

    @staticmethod
    def min_index(array: np.ndarray):
        k = array.argmin()
        w = array.shape[1]
        return k % w, k // w

    def assign_pixels_to_clusters(self):
        alpha = self.alpha
        for cluster in self.clusters:
            cluster: Cluster
            h_low = cluster.h_low
            w_low = cluster.w_low
            h_high = cluster.h_high
            w_high = cluster.w_high

            l_part = self.data[h_low:h_high, w_low:w_high, 0]
            a_part = self.data[h_low:h_high, w_low:w_high, 1]
            b_part = self.data[h_low:h_high, w_low:w_high, 2]

            # calculate color distance
            color_distance = np.sqrt(((l_part - cluster.l) ** 2 +
                                      (a_part - cluster.a) ** 2 +
                                      (b_part - cluster.b) ** 2))

            width_part = self.x_coordinate[h_low:h_high, w_low:w_high]
            height_part = self.y_coordinate[h_low:h_high, w_low:w_high]

            # calculate spatial distance
            spatial_distance = np.sqrt((height_part - cluster.h) ** 2 +
                                       (width_part - cluster.w) ** 2)

            # calculate total distance
            total_distance = (1 - alpha) * color_distance + (alpha / self.S) * spatial_distance

            mask = (total_distance < self.distance[h_low:h_high, w_low:w_high])

            self.distance[h_low:h_high, w_low:w_high][mask] = total_distance[mask]
            self.pixel_to_cluster[h_low:h_high, w_low:w_high][mask] = cluster.index

        self.distance = np.full((self.image_height, self.image_width), np.inf)

    def update_cluster_centers(self):
        for cluster in self.clusters:
            h_low = cluster.h_low
            w_low = cluster.w_low
            h_high = cluster.h_high
            w_high = cluster.w_high

            mask = self.pixel_to_cluster[h_low:h_high, w_low:w_high] == cluster.index

            # number of pixels intended to this cluster
            count = np.count_nonzero(mask)

            # calculate row and col for each pixel intended to this cluster
            rows, cols = np.where(mask)
            rows += h_low
            cols += w_low

            if count == 0:
                continue

            # calculate new cluster center
            h_mean = rows.sum() // count
            w_mean = cols.sum() // count

            # update cluster parameters based on new cluster center
            cluster.update(h_mean,
                           w_mean,
                           self.data[h_mean][w_mean][0],
                           self.data[h_mean][w_mean][1],
                           self.data[h_mean][w_mean][2])

    def compute_gradient_magnitude(self):
        gradiant_magnitude = np.zeros((self.image_height, self.image_width)).astype('float64')

        for channel in range(3):
            # calculate gradiant in x and y direction
            x_gradiant = nd.sobel(self.data[:, :, channel], axis=0)
            y_gradiant = nd.sobel(self.data[:, :, channel], axis=1)
            # calculate gradiant magnitude
            gradiant_magnitude += (x_gradiant ** 2 + y_gradiant ** 2)

        return gradiant_magnitude

    def compute_segmented_image(self):
        # mark boundaries
        self.copy = (sm.mark_boundaries(self.copy,
                                        self.pixel_to_cluster,
                                        color=self.color) * 255).astype('uint8')

        # mark cluster centers
        thickness = 2
        for cluster in self.clusters:
            h, w = cluster.h, cluster.w
            self.copy[h - thickness:h + thickness, w - thickness:w + thickness, 0] = 255
            self.copy[h - thickness:h + thickness, w - thickness:w + thickness, 1] = 255
            self.copy[h - thickness:h + thickness, w - thickness:w + thickness, 2] = 255

        return self.copy

    def iteration(self):
        log.info('clusters initialised ...')
        # initialise clusters
        self.initialise_clusters()

        log.info('clusters moved briefly ...')
        # move clusters based on gradiant magnitude value
        self.initial_move_clusters()

        log.info('main process started ...')
        for i in trange(self.max_iteration):
            # assign pixels to closest cluster based on total distance value
            self.assign_pixels_to_clusters()

            # update cluster centers based on new pixel to cluster array
            self.update_cluster_centers()

        log.info('main process done ...')

        if self.post_process:
            log.info('connected components started ...')

            # calculate kd_tree data
            centers = [[cluster.w, cluster.h] for cluster in self.clusters]
            self.kd_tree = kd.KDTree(centers)
            for i in trange(1):
                self.connected_component_post_process()

            log.warning('connected components done ...')

        # compute segmented image
        self.compute_segmented_image()

    def save(self):
        # set image path
        path = 'output/{filename}-alpha{alpha}-K={k}-pre_smooth_std={std}.jpg'.format(std=self.std,
                                                                                      alpha=self.alpha,
                                                                                      k=self.K,
                                                                                      filename=self.filename)
        # maximize image (if pyrdown method used)
        # self.copy = cv.pyrUp(self.copy)

        # save image
        io.imsave(path, self.copy)

    def connected_component_post_process(self):

        # maximum distance between pixel and closest cluster center
        max_dis = int(self.S * math.sqrt(2))

        for cluster in self.clusters:
            h_low, h_high = cluster.h_low, cluster.h_high
            w_low, w_high = cluster.w_low, cluster.w_high
            # get cluster area
            current_area = self.pixel_to_cluster[h_low:h_high, w_low:w_high]

            # compute pixels intended to cluster
            mask = (current_area == cluster.index).astype('uint8')

            # compute connected components
            components_number, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

            if components_number >= 3:
                # sort by area size (descending)
                sorted_labels_by_area_size = np.argsort(-stats[:, -1])
                for i in range(2, len(sorted_labels_by_area_size)):
                    # connected component label
                    component_label = sorted_labels_by_area_size[i]

                    # find connected component pixels
                    mask = labels == component_label

                    # find connected component pixels (row, col)
                    rows, cols = np.where(mask)
                    rows += h_low
                    cols += w_low
                    cols_rows = np.array((cols, rows)).T

                    # compute closest cluster center per pixel in connected component
                    _, indices = self.kd_tree.query(cols_rows, k=1, distance_upper_bound=max_dis)
                    cluster_centers = self.kd_tree.data[indices]
                    cluster_indices = self.pixel_to_cluster[cluster_centers[:, 1], cluster_centers[:, 0]]

                    # set intended cluster index to pixel
                    self.pixel_to_cluster[rows, cols] = cluster_indices


def main():
    # set logger configuration
    log.basicConfig(level=log.INFO)

    # read image
    input_image_file_name = 'input/lenna.png'
    image_name = 'lenna'
    image = io.imread(input_image_file_name)

    # minimize image (higher speed)
    # image = cv.pyrDown(image)

    # set initial parameters
    alpha, k, std = .98, 512, .25

    # instantiate slic_processor
    slic_processor = SLICProcessor(input_image=image,
                                   filename=image_name,
                                   alpha=alpha,
                                   K=k,
                                   std=std,
                                   rgb_color=(0, 0, 0),
                                   post_process=True,
                                   max_iteration=30)

    # iteration method
    slic_processor.iteration()

    # compute segmented image
    segmented_image = slic_processor.compute_segmented_image()

    # save image
    slic_processor.save()


if __name__ == '__main__':
    main()
