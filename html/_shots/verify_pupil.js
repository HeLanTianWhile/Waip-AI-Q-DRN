const c = document.querySelector('.stage').getBoundingClientRect();
const x = c.left + c.width / 2 + 260;  // 鼠标在圆心右下方
const y = c.top + c.height / 2 + 180;
window.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: y }));
await new Promise(r => setTimeout(r, 700));  // 等 rAF 插值稳定
const pupil = document.querySelector('.pupil');
const w0 = document.querySelector('.w');
return {
  pupilTransform: getComputedStyle(pupil).transform,
  w0Transform: getComputedStyle(w0).transform
};
