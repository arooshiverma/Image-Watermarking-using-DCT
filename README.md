# Image-Watermarking-using-DCT
Uses Discrete Cosine Transform(DCT) only. This is an invisible watermarking scheme. The algorithm also computes Normalized correlation between the actual and extracted watermark images. We also test a variety of geometric and signal-based attacks.

## Embedding Watermark
The function watermark_image(img, wm) embeds the watermark wm into image img.

The watermarking algorithm is a pretty simple one:
1. Divide the image into non-overlapping blocks of dimensions 8x8. Exclude the edge b_cut(50 in our example) pixels from the edges.
2. Let’s assume we have n such blocks
3. Let the value of key = k. This will be used as a seed for the pseudorandom generator.
4. Let the dimensions of the watermark be wxh. Thus there are pix = w*h pixels in the watermark. We convert the watermark to a black and white, one channel image.
5. Check if n>=pix. If no, then the watermark cannot be embedded.
6. Create an empty set st.
7. We repeat the following steps for each pixel of the watermark:\
      a. Generate the next pseudorandom number = y until you get a number that is not in the set.\
      b. Compute the DCT of this block. Let it be named dct_block.\
      c. Let elem = dct_block[0][0].\
      d. Divide elem by a number fact, which is defined as 8 for our case.\
      e. Now, if the pixel has value 255, round off elem to the nearest odd number. If the pixel is 0, round it off to the nearest even number.\
      f. Multiply elem by fact.\
      g. Store this value of elem in dct_block[0][0].\
      h. Compute Inverse DCT of the block and store it in the appropriate position of the watermarked image.
8. Save the Watermarked Image.

![DCT block](https://github.com/arooshiverma/Image-Watermarking-using-DCT/blob/main/imgs/img1.JPG?raw=true)

## Extracting Watermark
The function extract_watermark(img, ext_name) extracts the watermark from image img and saves it to a file with name 'ext_name'.

The watermark extraction algorithm:
1. Divide the watermarked image into non-overlapping blocks of dimensions
8x8. Exclude the edge b_cut(50 in our example) pixels from the edges.
2. Let’s assume we have n such blocks
3. Let the value of key = k. This will be used as a seed for the pseudorandom generator.
4. The dimensions of the watermark are fixed to 64x64 in our case.
5. Create an empty set st.
6. We repeat the following steps for each pixel of the watermark:\
    a. Generate the next pseudorandom number = y until you get a number that is not in the set.\
    b. Compute the DCT of this block. Let it be named dct_block.\
    c. Let elem = dct_block[0][0].\
    d. Divide elem by a number fact, which is defined as 8 for our case.\
    e. If the closest integer to this element is odd, the particular watermark pixel is 255. Else, it is 0.
7. Thus, we get the watermark 2d array. Save this image and calculate Normalized Cross-Correlation (NC) index.


## Note
* Why the division by fact? We use python for the computation of DCT. Then we make changes to this DCT and compute final = IDCT(DCT). Right now, the final contains floating-point values. But when we save it as an image, the floating-point numbers are converted into integers. Thus, when we read the image during extraction, the computed DCT has slight variations. To solve this, we divide the DCT coefficient by fact, change it to even/odd as required, and multiply by fact. Thus, now DCT values can have a variance from -fact/2 to +fact/2, and our result won’t change.
* It is pretty easy to attack this algorithm. A person can change all block’s dct[0][0]’s value from odd to even or even to odd, randomly. This would essentially ruin the watermark.


## Attacks
### Geometric attacks
* Scaling to half
* Scaling to bigger
* Cutting 100 rows
### Signal-based attacks
* Average filter
* Median filter
* Gaussian noise 
* Salt & Pepper noise
* Speckle Noise
