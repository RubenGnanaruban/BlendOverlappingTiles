# Blend Overlapping Tiles

This code generates a composite image by stitching smaller images in a grid pattern. 

## When do I need this?
When converting images using a generative AI model, the size of the image is limited by the available vRAM. Therefore a bigger image is cropped and the small FOVs are converted. This is generally a good solution, except the adjacent images do not always match at the boundaries. In order to overcome this issue, we make use of overlapping tiles to get rid of the grid pattern in the final result. 
