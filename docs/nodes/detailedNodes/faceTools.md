# Face Nodes

## FaceOff

FaceOff mimics a user finding a face in an image and resizing the bounding box
around the head in Canvas.

Enter a face ID (found with FaceIdentifier) to choose which face to mask.

Just as you would add more context inside the bounding box by making it larger
in Canvas, the node gives you a padding input (in pixels) which will
simultaneously add more context, and increase the resolution of the bounding box
so the face remains the same size inside it.

The "Minimum Confidence" input defaults to 0.5 (50%), and represents a pass/fail
threshold a detected face must reach for it to be processed. Lowering this value
may help if detection is failing. If the detected masks are imperfect and stray
too far outside/inside of faces, the node gives you X & Y offsets to shrink/grow
the masks by a multiplier.

FaceOff will output the face in a bounded image, taking the face off of the
original image for input into any node that accepts image inputs. The node also
outputs a face mask with the dimensions of the bounded image. The X & Y outputs
are for connecting to the X & Y inputs of the Paste Image node, which will place
the bounded image back on the original image using these coordinates.

###### Inputs/Outputs

| Input              | Description                                                                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image              | Image for face detection                                                                                                                                                                 |
| Face ID            | The face ID to process, numbered from 0. Multiple faces not supported. Find a face's ID with FaceIdentifier node.                                                                        |
| Minimum Confidence | Minimum confidence for face detection (lower if detection is failing)                                                                                                                    |
| X Offset           | X-axis offset of the mask                                                                                                                                                                |
| Y Offset           | Y-axis offset of the mask                                                                                                                                                                |
| Padding            | All-axis padding around the mask in pixels                                                                                                                                               |
| Chunk              | Chunk (or divide) the image into sections to greatly improve face detection success. Defaults to off, but will activate if no faces are detected normally. Activate to chunk by default. |

| Output        | Description                                      |
| ------------- | ------------------------------------------------ |
| Bounded Image | Original image bound, cropped, and resized       |
| Width         | The width of the bounded image in pixels         |
| Height        | The height of the bounded image in pixels        |
| Mask          | The output mask                                  |
| X             | The x coordinate of the bounding box's left side |
| Y             | The y coordinate of the bounding box's top side  |

## FaceMask

FaceMask mimics a user drawing masks on faces in an image in Canvas.

The "Face IDs" input allows the user to select specific faces to be masked.
Leave empty to detect and mask all faces, or a comma-separated list for a
specific combination of faces (ex: `1,2,4`). A single integer will detect and
mask that specific face. Find face IDs with the FaceIdentifier node.

The "Minimum Confidence" input defaults to 0.5 (50%), and represents a pass/fail
threshold a detected face must reach for it to be processed. Lowering this value
may help if detection is failing.

If the detected masks are imperfect and stray too far outside/inside of faces,
the node gives you X & Y offsets to shrink/grow the masks by a multiplier. All
masks shrink/grow together by the X & Y offset values.

By default, masks are created to change faces. When masks are inverted, they
change surrounding areas, protecting faces.

###### Inputs/Outputs

| Input              | Description                                                                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image              | Image for face detection                                                                                                                                                                 |
| Face IDs           | Comma-separated list of face ids to mask eg '0,2,7'. Numbered from 0. Leave empty to mask all. Find face IDs with FaceIdentifier node.                                                   |
| Minimum Confidence | Minimum confidence for face detection (lower if detection is failing)                                                                                                                    |
| X Offset           | X-axis offset of the mask                                                                                                                                                                |
| Y Offset           | Y-axis offset of the mask                                                                                                                                                                |
| Chunk              | Chunk (or divide) the image into sections to greatly improve face detection success. Defaults to off, but will activate if no faces are detected normally. Activate to chunk by default. |
| Invert Mask        | Toggle to invert the face mask                                                                                                                                                           |

| Output | Description                       |
| ------ | --------------------------------- |
| Image  | The original image                |
| Width  | The width of the image in pixels  |
| Height | The height of the image in pixels |
| Mask   | The output face mask              |

## FaceIdentifier

FaceIdentifier outputs an image with detected face IDs printed in white numbers
onto each face.

Face IDs can then be used in FaceMask and FaceOff to selectively mask all, a
specific combination, or single faces.

The FaceIdentifier output image is generated for user reference, and isn't meant
to be passed on to other image-processing nodes.

The "Minimum Confidence" input defaults to 0.5 (50%), and represents a pass/fail
threshold a detected face must reach for it to be processed. Lowering this value
may help if detection is failing. If an image is changed in the slightest, run
it through FaceIdentifier again to get updated FaceIDs.

###### Inputs/Outputs

| Input              | Description                                                                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image              | Image for face detection                                                                                                                                                                 |
| Minimum Confidence | Minimum confidence for face detection (lower if detection is failing)                                                                                                                    |
| Chunk              | Chunk (or divide) the image into sections to greatly improve face detection success. Defaults to off, but will activate if no faces are detected normally. Activate to chunk by default. |

| Output | Description                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------ |
| Image  | The original image with small face ID numbers printed in white onto each face for user reference |
| Width  | The width of the original image in pixels                                                        |
| Height | The height of the original image in pixels                                                       |

## Tips

- If not all target faces are being detected, activate Chunk to bypass full
  image face detection and greatly improve detection success.
- Final results will vary between full-image detection and chunking for faces
  that are detectable by both due to the nature of the process. Try either to
  your taste.
- Be sure Minimum Confidence is set the same when using FaceIdentifier with
  FaceOff/FaceMask.
- For FaceOff, use the color correction node before faceplace to correct edges
  being noticeable in the final image (see example screenshot).
- Non-inpainting models may struggle to paint/generate correctly around faces.
- If your face won't change the way you want it to no matter what you change,
  consider that the change you're trying to make is too much at that resolution.
  For example, if an image is only 512x768 total, the face might only be 128x128
  or 256x256, much smaller than the 512x512 your SD1.5 model was probably
  trained on. Try increasing the resolution of the image by upscaling or
  resizing, add padding to increase the bounding box's resolution, or use an
  image where the face takes up more pixels.
- If the resulting face seems out of place pasted back on the original image
  (ie. too large, not proportional), add more padding on the FaceOff node to
  give inpainting more context. Context and good prompting are important to
  keeping things proportional.
- If you find the mask is too big/small and going too far outside/inside the
  area you want to affect, adjust the x & y offsets to shrink/grow the mask area
- Use a higher denoise start value to resemble aspects of the original face or
  surroundings. Denoise start = 0 & denoise end = 1 will make something new,
  while denoise start = 0.50 & denoise end = 1 will be 50% old and 50% new.
- mediapipe isn't good at detecting faces with lots of face paint, hair covering
  the face, etc. Anything that obstructs the face will likely result in no faces
  being detected.
- If you find your face isn't being detected, try lowering the minimum
  confidence value from 0.5. This could result in false positives, however
  (random areas being detected as faces and masked).
- After altering an image and wanting to process a different face in the newly
  altered image, run the altered image through FaceIdentifier again to see the
  new Face IDs. MediaPipe will most likely detect faces in a different order
  after an image has been changed in the slightest.
