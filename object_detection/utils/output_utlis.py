def convert_output_to_detections(output_dict, category_index, threshold, width,
                                 height):
    """Converts the output detections of the Tensorflow Object API
    to a list of dictionairies of detections. Converts the class to
    the classname.

    :output_dict: output of detector
    :category_index: dictionairy of categories with their index
    :threshold: threshold for accuracy
    :width: width of the image
    :height: height of the image
    :returns: list of dictionairies of the filtered detections

    """
    detections = list()

    for index in range(len(output_dict['detection_boxes'])):
        score = output_dict['detection_scores'][index]
        if score > threshold:
            class_name = category_index[output_dict['detection_classes']
                                        [index]]['name']
            detection = {
                "class": class_name,
                "score": output_dict['detection_scores'][index],
                "box": (
                    int(width * output_dict['detection_boxes'][index][1]),
                    int(height * output_dict['detection_boxes'][index][0]),
                    int(width * output_dict['detection_boxes'][index][3]),
                    int(height * output_dict['detection_boxes'][index][2])
                )
            }
            detections.append(detection)
        else:
            return detections
    return detections
