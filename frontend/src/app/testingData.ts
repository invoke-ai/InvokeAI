// images for testing only
import testImage1 from '../assets/images/test1.png';
import testImage2 from '../assets/images/test2.png';
import testImage3 from '../assets/images/test3.png';
import testImage4 from '../assets/images/test4.png';
import testImage5 from '../assets/images/test5.png';
import testImage6 from '../assets/images/test6.png';
import testImage7 from '../assets/images/test7.png';
import testImage8 from '../assets/images/test8.png';
import testImage9 from '../assets/images/test9.png';
import testImage10 from '../assets/images/test10.png';
import { SDImage } from './sdSlice';
import { v4 as uuidv4 } from 'uuid';

// populate gallery for testing
export const testImages: Array<SDImage> = [
  testImage1,
  testImage2,
  testImage3,
  testImage4,
  testImage5,
  testImage6,
  testImage7,
  testImage8,
  testImage9,
  testImage10,
].map((url, i) => {
  return {
    uuid: uuidv4(),
    url: url,
    metadata: {
      prompt: `test image ${i} prompt`,
    },
  };
});

export const testLogs: Array<string> =
  `127.0.0.1 - - [04/Sep/2022 15:31:35] "GET /outputs/img-samples/000034.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 15:32:08] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 34/34 [00:39<00:00,  1.17s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:41<00:00, 41.23s/it]
>> Usage stats:
>>   1 image(s) generated in 41.23s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 15:32:49] "GET /outputs/img-samples/000035.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 15:32:56] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 34/34 [00:40<00:00,  1.21s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:42<00:00, 42.84s/it]
>> Usage stats:
>>   1 image(s) generated in 42.84s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 15:33:39] "GET /outputs/img-samples/000036.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 16:54:40] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
>> Setting Sampler to k_euler_a
100%|███████████████████████████████████████████| 15/15 [00:21<00:00,  1.44s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:24<00:00, 24.17s/it]
>> Usage stats:
>>   1 image(s) generated in 24.17s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 16:55:05] "GET /outputs/img-samples/000037.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 16:55:12] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 22/22 [00:26<00:00,  1.21s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:28<00:00, 28.55s/it]
>> Usage stats:
>>   1 image(s) generated in 28.55s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 16:55:41] "GET /outputs/img-samples/000038.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 16:56:38] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 22/22 [00:25<00:00,  1.14s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:27<00:00, 27.21s/it]
>> Usage stats:
>>   1 image(s) generated in 27.21s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 16:57:05] "GET /outputs/img-samples/000039.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 16:57:22] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 23/23 [00:24<00:00,  1.09s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:26<00:00, 26.61s/it]
>> Usage stats:
>>   1 image(s) generated in 26.61s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 16:57:48] "GET /outputs/img-samples/000040.2413046150.png HTTP/1.1" 200 -
127.0.0.1 - - [04/Sep/2022 16:57:51] "POST / HTTP/1.1" 200 -
>> Request to generate with prompt: Caveman
100%|███████████████████████████████████████████| 23/23 [00:25<00:00,  1.11s/it]
Generating: 100%|█████████████████████████████████| 1/1 [00:26<00:00, 26.99s/it]
>> Usage stats:
>>   1 image(s) generated in 26.99s
>>   Max VRAM used for this generation: 0.00G
127.0.0.1 - - [04/Sep/2022 16:58:18] "GET /outputs/img-samples/000041.2413046150.png HTTP/1.1" 200 -`.split(
    '\n'
  );
