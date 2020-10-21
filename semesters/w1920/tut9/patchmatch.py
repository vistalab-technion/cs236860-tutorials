import numpy as np
import cv2
import time

SCALE = 1

class PatchMatch:
    ALPHA = 0.5
    DEFAULT_PATCH_SIZE = 7
    DEFAULT_NUM_ITERATIONS = 5

    def _init_nnf(self):
        original_i, original_j = np.meshgrid(
            np.arange(self._source.shape[0]),
            np.arange(self._source.shape[1]),
            indexing='ij'
        )

        self._nnf_i = np.random.randint(target.shape[0], size=source.shape[:2]) - original_i
        self._nnf_j = np.random.randint(target.shape[1], size=source.shape[:2]) - original_j

    def _get_target_indices(self, i, j):
        target_i = i + self._nnf_i[i, j]
        target_j = j + self._nnf_j[i, j]
        return target_i, target_j

    def _calculate_distance(self, i, j, target_i, target_j):
        dx_left = min(j, target_j, self._patch_radius)
        dx_right = min(self._source.shape[1] - j, self._target.shape[1] - target_j, self._patch_radius + 1)
        dy_up = min(i, target_i, self._patch_radius)
        dy_down = min(self._source.shape[0] - i, self._target.shape[0] - target_i, self._patch_radius + 1)

        source_patch = self._source[i - dy_up:i + dy_down, j - dx_left:j + dx_right]
        target_patch = self._target[target_i - dy_up:target_i + dy_down, target_j - dx_left:target_j + dx_right]
        return np.linalg.norm(source_patch - target_patch) ** 2 / (dx_left + dx_right) / (dy_up + dy_down)

    def _init_distances(self):
        self._distances = np.empty(self._source.shape[:2])
        for i in range(self._source.shape[0]):
            for j in range(self._source.shape[1]):
                target_i, target_j = self._get_target_indices(i, j)
                self._distances[i, j] = self._calculate_distance(i, j, target_i, target_j)

    def _clip_offset(self, i, j, offset_i, offset_j):
        offset_i = np.clip(offset_i, -i, self._target.shape[0] - 1 - i)
        offset_j = np.clip(offset_j, -j, self._target.shape[1] - 1 - j)
        return offset_i, offset_j

    def _update_nnf(self, i, j, offset_i, offset_j):
        offset_i, offset_j = self._clip_offset(i, j, offset_i, offset_j)
        new_distance = self._calculate_distance(i, j, i + offset_i, j + offset_j)
        if new_distance < self._distances[i, j]:
            self._nnf_i[i, j] = offset_i
            self._nnf_j[i, j] = offset_j
            self._distances[i, j] = new_distance

    def _propagate(self, i, j, reverse=False):
        direction = 1 if reverse else -1

        neighbor_i = i + direction
        if 0 <= neighbor_i < self._source.shape[0]:
            self._update_nnf(i, j, self._nnf_i[neighbor_i, j], self._nnf_j[neighbor_i, j])

        neighbor_j = j + direction
        if 0 <= neighbor_j < self._source.shape[1]:
            self._update_nnf(i, j, self._nnf_i[i, neighbor_j], self._nnf_j[i, neighbor_j])

    def _random_search(self, i, j):
        original_offset_i = self._nnf_i[i, j]
        original_offset_j = self._nnf_j[i, j]

        search_radius = self._search_radius
        while search_radius > 0:
            min_offset_i, min_offset_j = self._clip_offset(i, j, original_offset_i - search_radius, original_offset_j - search_radius)
            max_offset_i, max_offset_j = self._clip_offset(i, j, original_offset_i + search_radius, original_offset_j + search_radius)
            offset_i = np.random.randint(low=min_offset_i, high=max_offset_i)
            offset_j = np.random.randint(low=min_offset_j, high=max_offset_j)
            self._update_nnf(i, j, offset_i, offset_j)
            search_radius = np.floor(search_radius * type(self).ALPHA)

    def __init__(self, source, target, patch_size=DEFAULT_PATCH_SIZE):
        self._source = source
        self._target = target
        self._patch_radius = patch_size // 2
        self._search_radius = max(*self._target.shape[:2]) // 2
        self._init_nnf()
        self._init_distances()

    def improve(self, reverse=False):
        i_order = np.arange(self._source.shape[0])
        j_order = np.arange(self._source.shape[1])
        if reverse:
            i_order = np.flip(i_order)
            j_order = np.flip(j_order)

        for i in i_order:
            for j in j_order:
                self._propagate(i, j, reverse=reverse)
                self._random_search(i, j)

    def solve(self, num_iterations=DEFAULT_NUM_ITERATIONS):
        reverse = False
        for _ in range(num_iterations):
            self.improve(reverse=reverse)
            reverse = not reverse

    def reconstruct(self):
        image = np.empty(self._source.shape)
        for i in range(self._source.shape[0]):
            for j in range(self._source.shape[1]):
                target_i, target_j = self._get_target_indices(i, j)
                image[i, j] = self._target[target_i, target_j]

        return image

if __name__ == "__main__":
    source = cv2.imread('view1.png')
    target = cv2.imread('view5.png')
    source = cv2.resize(source, None, fx=SCALE, fy=SCALE)
    target = cv2.resize(target, None, fx=SCALE, fy=SCALE)

    print('started')
    start = time.time()
    match = PatchMatch(source, target)
    match.solve()
    result = match.reconstruct()
    cv2.imwrite('result.png', result)
    end = time.time()
    print('finished (%s seconds)' % (end - start, ))
