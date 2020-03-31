def LabelDecoder(labels, image_h, image_w) :
    list_rect = []

    for batch in range(labels.shape[0]):
        label = labels[batch]
        for grid_y in range(labels.shape[1]) :
            for grid_x in range(labels.shape[2]) :
                if(label[grid_y][grid_x][0] == 0) :
                    continue

                scale_factor = (1 / 7)
                center_x = (label[grid_y][grid_x][1] + grid_x) * scale_factor
                center_y = (label[grid_y][grid_x][2] + grid_y) * scale_factor

                w = label[grid_y][grid_x][3] * image_w
                h = label[grid_y][grid_x][4] * image_h

                center_x = center_x * image_w
                center_y = center_y * image_h

                x = center_x - (w / 2.0)
                y = center_y - (h / 2.0)

                x = int(round(x))
                y = int(round(y))
                w = int(round(w))
                h = int(round(h))

                list_rect.append([x, y, w, h])

    return list_rect