import { crc32, deflateSync } from 'node:zlib';

// PNG truecolour, filter 0. A finite palette of dimensions is shared by all seeded images.
const cache = new Map();
const chunk = (type, data) => {
  const body = Buffer.concat([Buffer.from(type), data]);
  const result = Buffer.alloc(body.length + 8);
  result.writeUInt32BE(data.length);
  body.copy(result, 4);
  result.writeUInt32BE(crc32(body), result.length - 4);
  return result;
};
export const mediaDimensions = ({ width, height }, thumbnail = false) => {
  const scale = thumbnail ? Math.min(1, 256 / width, 256 / height) : 1;
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
};
export const fixturePng = (width, height) => {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > 4096 ||
    height > 4096
  ) {
    throw new Error(`Invalid fixture image dimensions: ${width}x${height}`);
  }
  const key = `${width}x${height}`;
  if (cache.has(key)) {
    return cache.get(key);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  const pixels = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = y * (width * 3 + 1) + 1 + x * 3;
      const light = (Math.floor(x / 32) + Math.floor(y / 32)) % 2 === 0;
      pixels[offset] = light ? 64 : 192;
      pixels[offset + 1] = light ? 160 : 80;
      pixels[offset + 2] = 128;
    }
  }
  const bytes = Buffer.concat([
    Buffer.from('89504e470d0a1a0a', 'hex'),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(pixels)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  // Dynamic uploads may introduce dimensions; never retain an unbounded test-server cache.
  if (cache.size < 32) {
    cache.set(key, bytes);
  }
  return bytes;
};
