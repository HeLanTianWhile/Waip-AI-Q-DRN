const { app, BrowserWindow, Tray, Menu, screen, globalShortcut, desktopCapturer, ipcMain } = require('electron');
const path = require('path');

let win = null;        // 桌宠主窗口
let tray = null;
let controlWin = null; // 尺寸控制面板
const BASE_SIZE = 340;

function createWindow() {
  win = new BrowserWindow({
    width: BASE_SIZE,
    height: BASE_SIZE,
    transparent: true,
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    title: 'Waip AI',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.loadFile('pet.html');
  win.setAlwaysOnTop(true, 'screen-saver');

  // 轮询全局鼠标位置，换算成相对窗口中心的偏移，发给渲染进程
  setInterval(() => {
    if (!win || win.isDestroyed()) return;
    const p = screen.getCursorScreenPoint();
    const b = win.getBounds();
    win.webContents.send('mouse', {
      dx: p.x - (b.x + b.width / 2),
      dy: p.y - (b.y + b.height / 2)
    });
  }, 16);

  // 周期捕获屏幕缩略图 + 窗口位置，供渲染进程做 w 明暗检测
  setInterval(() => {
    if (!win || win.isDestroyed()) return;
    captureAndSend();
  }, 500);
}

async function captureAndSend() {
  try {
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: 320, height: 180 }
    });
    const src = sources && sources[0];
    if (!src) return;
    const display = screen.getPrimaryDisplay();
    const b = display.bounds;
    const wb = win.getBounds();
    const ts = src.thumbnail.getSize();
    win.webContents.send('screen', {
      image: src.thumbnail.toDataURL(),
      displayX: b.x, displayY: b.y, displayW: b.width, displayH: b.height,
      thumbW: ts.width, thumbH: ts.height,
      winX: wb.x, winY: wb.y
    });
  } catch (e) { /* 捕获失败则 w 保持默认黑色 */ }
}

function createControlWindow() {
  if (controlWin && !controlWin.isDestroyed()) { controlWin.show(); controlWin.focus(); return; }
  controlWin = new BrowserWindow({
    width: 280,
    height: 84,
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    transparent: false,
    title: 'Waip AI 尺寸',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  controlWin.loadFile('control.html');
  controlWin.on('closed', () => { controlWin = null; });
}

function resizePet(v) {
  if (!win || win.isDestroyed()) return;
  v = Math.max(180, Math.min(520, Math.round(v / 10) * 10 || BASE_SIZE));
  const b = win.getBounds();
  win.setBounds({
    x: b.x + Math.round((b.width - v) / 2),
    y: b.y + Math.round((b.height - v) / 2),
    width: v,
    height: v
  });
  win.webContents.send('scale', v / BASE_SIZE);
}

function createTray() {
  const { nativeImage } = require('electron');
  const icon = nativeImage.createFromPath(path.join(__dirname, 'tray.png'));
  if (icon.isEmpty()) return;
  tray = new Tray(icon);
  tray.setToolTip('Waip AI');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: '窗口置顶', type: 'checkbox', checked: true, click: (item) => {
        if (win && !win.isDestroyed()) win.setAlwaysOnTop(item.checked, 'screen-saver');
      } },
    { label: '调整大小…', click: () => createControlWindow() },
    { type: 'separator' },
    { label: '退出', click: () => app.quit() }
  ]));
  tray.on('click', () => { if (win && !win.isDestroyed()) { win.show(); } });
}

app.whenReady().then(() => {
  app.setName('Waip AI');
  createWindow();
  createTray();

  ipcMain.on('resize', (_e, v) => resizePet(v));

  globalShortcut.register('Ctrl+Alt+Q', () => app.quit());
  app.on('will-quit', () => globalShortcut.unregisterAll());
});

app.on('window-all-closed', () => {});
