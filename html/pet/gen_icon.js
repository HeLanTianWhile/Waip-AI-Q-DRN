// 生成托盘图标 tray.png（32x32 眼睛图案），用 Node 原生 zlib 手写 PNG
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');

const W = 32, H = 32;
const px = Buffer.alloc(H * (1 + W * 4));
const cx = W / 2 - 0.5, cy = H / 2 - 0.5;

for (let y = 0; y < H; y++) {
  px[y * (1 + W * 4)] = 0; // filter: none
  for (let x = 0; x < W; x++) {
    const d = Math.hypot(x - cx, y - cy);
    const i = y * (1 + W * 4) + 1 + x * 4;
    let r = 0, g = 0, b = 0, a = 0;
    // 中性占位图标：白色圆底 + 灰色圆（非眼睛图案），等用户提供正式图标
    if (d < 14) { r = 255; g = 255; b = 255; a = 255; }
    if (d < 9)  { r = 120; g = 120; b = 120; a = 255; }
    px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = a;
  }
}

function crc32(buf) {
  const table = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    table[n] = c;
  }
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >>> 8);
  return (crc ^ 0xFFFFFFFF) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
ihdr[8] = 8; ihdr[9] = 6; // 8bit RGBA

const png = Buffer.concat([
  sig,
  chunk('IHDR', ihdr),
  chunk('IDAT', zlib.deflateSync(px)),
  chunk('IEND', Buffer.alloc(0))
]);
fs.writeFileSync(path.join(__dirname, 'tray.png'), png);
console.log('tray.png generated');
