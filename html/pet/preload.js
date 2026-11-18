const { contextBridge, ipcRenderer } = require('electron');

// 主桌宠窗口
contextBridge.exposeInMainWorld('pet', {
  onMouse: (cb) => ipcRenderer.on('mouse', (_e, d) => cb(d)),
  onScreen: (cb) => ipcRenderer.on('screen', (_e, d) => cb(d)),
  onScale: (cb) => ipcRenderer.on('scale', (_e, s) => cb(s))
});

// 尺寸控制面板窗口
contextBridge.exposeInMainWorld('control', {
  sendResize: (v) => ipcRenderer.send('resize', v)
});
