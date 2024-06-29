# Python module that defines script to processs input image for the FeMaSR model and perform the Super-Resolution
import math
import numpy as np

def predictor(session, inputs, H, W):
    if H * W >= 600 ** 2:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = inputs.shape
        output_height = height * 4
        output_width = width * 4
        output_shape = (batch, channel, output_height, output_width)
        tile_size = 240
        tile_pad = 16
        # start with black image
        output = np.zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = inputs[:, :, input_start_y_pad : input_end_y_pad, input_start_x_pad : input_end_x_pad]

                # upscale tile
                output_name = session.get_outputs()[0].name
                output_tile = session.run([output_name], {input : input_tile})[0]

                # output tile area on total image
                output_start_x = input_start_x * 4
                output_end_x = input_end_x * 4
                output_start_y = input_start_y * 4
                output_end_y = input_end_y * 4

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                output_end_x_tile = output_start_x_tile + input_tile_width *4
                output_start_y_tile = (input_start_y - input_start_y_pad) *4
                output_end_y_tile = output_start_y_tile + input_tile_height * 4

                # put tile into output image
                output[:, :, output_start_y : output_end_y,
                    output_start_x : output_end_x] = output_tile[:, :, output_start_y_tile : output_end_y_tile,
                                                                output_start_x_tile : output_end_x_tile]
        out_mat = output
        
    else:
        output_name = session.get_outputs()[0].name
        out_mat = session.run([output_name], {input : inputs})[0]
    out_mat = np.squeeze(out_mat, 0)
    out_mat = np.transpose(out_mat, (1, 2, 0))
    out_mat = out_mat * 255.0
    return out_mat