from PIL import Image, ImageDraw
import math

def calculate_points(distance_matrix):
    num_points = len(distance_matrix)
    center = (0, 0)
    points = [center]

    for i in range(1, num_points):
        angle = 2 * math.pi * i / (num_points - 1)
        dx = distance_matrix[0][i] * math.cos(angle)
        dy = distance_matrix[0][i] * math.sin(angle)
        point = (center[0] + dx, center[1] + dy)
        points.append(point)

    return points

def visualize_tsp_path(distance_matrix, path, output_filename=None):
    points = calculate_points(distance_matrix)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)

    padding = 50
    width = int(max_x - min_x + 2 * padding)
    height = int(max_y - min_y + 2 * padding)

    translated_points = [(x - min_x + padding, y - min_y + padding) for x, y in points]

    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for i, point1 in enumerate(translated_points):
        for j, point2 in enumerate(translated_points):
            if i != j:
                draw.line((point1, point2), fill=(128, 128, 128), width=1)

    for i in range(len(path) - 1):
        start_point = translated_points[path[i]]
        end_point = translated_points[path[i + 1]]
        draw.line((start_point, end_point), fill=(255, 0, 0), width=2)

    for idx, point in enumerate(translated_points):
        draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=(0, 0, 255), outline=(0, 0, 0))
        draw.text((point[0] + 10, point[1] - 10), str(idx), fill=(0, 0, 0))

    if output_filename is not None:
        image.save("./images/" + output_filename)
    image.show()

def displayTheGraph(distance_matrix, path, fileName):
    if fileName:
        # display and save file
        fileName = fileName + ".png"
        visualize_tsp_path(distance_matrix, path, output_filename=fileName)
    else:
        visualize_tsp_path(distance_matrix, path)