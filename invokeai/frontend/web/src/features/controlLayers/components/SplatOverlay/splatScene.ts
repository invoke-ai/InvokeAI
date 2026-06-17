import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper.js';

const FLOOR_Y = -0.5; // TripoSplat normalizes the object into ~[-0.5, 0.5]; its base rests on the floor here
const BG_COLOR = 0x3a3a40; // neutral studio grey, live preview only (capture is transparent)

/**
 * A grounded three.js + Spark viewport for a single 3D Gaussian splat: orbit/zoom/pan camera, a floor grid,
 * origin axes, and a clickable corner navigation gizmo (ViewHelper). The grid/axes/gizmo/background are
 * shown only in the live preview; on capture they're hidden and the background is transparent so the baked
 * layer is the object alone.
 *
 * The corner gizmo requires the renderer's `autoClear` to be off (it draws over the main render via a corner
 * viewport), so the main pass clears manually each frame.
 *
 * Orientation: TripoSplat .ply is -Y-up (3DGS). A parent group is flipped 180° about X so the object stands
 * upright (+Y up); the splat is yawed 90° so its front faces the camera.
 */
export class SplatScene {
  private readonly container: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly viewHelper: ViewHelper;
  private readonly clock: THREE.Clock;
  private readonly splatRoot: THREE.Group;
  private readonly grid: THREE.GridHelper;
  private readonly axes: THREE.AxesHelper;
  private readonly resizeObserver: ResizeObserver;
  private readonly onPointerDown: (e: PointerEvent) => void;
  private readonly onPointerUp: (e: PointerEvent) => void;
  private pointerDownPos: { x: number; y: number } | null = null;
  private mesh: SplatMesh | null = null;
  private disposed = false;

  constructor(container: HTMLElement) {
    this.container = container;
    this.clock = new THREE.Clock();

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    this.camera.position.set(0, 0, 2.2);

    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true, // required so capture() can read the rendered frame
    });
    this.renderer.autoClear = false; // ViewHelper draws over the main render; we clear manually
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.renderer.domElement.style.display = 'block';
    container.appendChild(this.renderer.domElement);

    const spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(spark);

    this.splatRoot = new THREE.Group();
    this.splatRoot.rotation.x = Math.PI;
    this.scene.add(this.splatRoot);

    this.grid = new THREE.GridHelper(6, 12, 0x8a8a8a, 0x555560);
    this.grid.position.y = FLOOR_Y;
    this.scene.add(this.grid);
    this.axes = new THREE.AxesHelper(0.4);
    this.axes.position.y = FLOOR_Y;
    this.scene.add(this.axes);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 50;
    this.controls.target.set(0, 0, 0);

    this.viewHelper = new ViewHelper(this.camera, this.renderer.domElement);
    this.viewHelper.center = this.controls.target; // snap orbits around the same point as OrbitControls

    // Only treat a near-stationary pointerup as a gizmo click (so orbiting doesn't accidentally snap).
    this.onPointerDown = (e) => {
      this.pointerDownPos = { x: e.clientX, y: e.clientY };
    };
    this.onPointerUp = (e) => {
      const down = this.pointerDownPos;
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
        this.viewHelper.handleClick(e);
      }
    };
    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);

    this.resizeObserver = new ResizeObserver(this.resize);
    this.resizeObserver.observe(container);
    this.resize();

    this.renderer.setAnimationLoop(this.tick);
  }

  private tick = (): void => {
    if (this.disposed) {
      return;
    }
    const delta = this.clock.getDelta();
    if (this.viewHelper.animating) {
      this.viewHelper.update(delta); // drive the camera during a snap; don't let OrbitControls fight it
    } else {
      this.controls.update();
    }
    this.renderer.setClearColor(BG_COLOR, 1);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
    this.viewHelper.render(this.renderer);
  };

  resize = (): void => {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  };

  /** Position the camera for a front-on default view that roughly fills the frame; user can orbit from there. */
  private frameObject(): void {
    const halfExtent = 0.5;
    const margin = 1.1;
    const aspect = this.camera.aspect || 1;
    const vfov = (this.camera.fov * Math.PI) / 180;
    const dv = (halfExtent * margin) / Math.tan(vfov / 2);
    const dist = Math.max(dv, dv / aspect);
    this.camera.position.set(0, 0, dist);
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  async loadFromUrl(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch splat: HTTP ${res.status}`);
    }
    const bytes = await res.arrayBuffer();
    if (this.disposed) {
      return;
    }
    const mesh = new SplatMesh({ fileBytes: bytes, fileName: url.split('/').pop() || 'model.ply' });
    await mesh.initialized;
    if (this.disposed) {
      this.disposeMesh(mesh);
      return;
    }
    if (this.mesh) {
      this.splatRoot.remove(this.mesh);
      this.disposeMesh(this.mesh);
    }
    mesh.rotation.y = Math.PI / 2;
    this.mesh = mesh;
    this.splatRoot.add(mesh);
    this.frameObject();
  }

  /**
   * Capture the object alone (no grid/axes/gizmo/background) at the given pixel size, from the user's current
   * camera (WYSIWYG). Renders at pixelRatio 1 so the blob's intrinsic size equals width×height. The live
   * render loop is paused during capture to avoid it overwriting the frame before toBlob reads it.
   */
  async capture(width: number, height: number): Promise<Blob | null> {
    this.renderer.setAnimationLoop(null);
    this.grid.visible = false;
    this.axes.visible = false;
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setClearColor(0x000000, 0); // transparent
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
    const blob = await new Promise<Blob | null>((resolve) => {
      this.renderer.domElement.toBlob(resolve, 'image/png');
    });
    // Restore the live preview.
    this.grid.visible = true;
    this.axes.visible = true;
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.resize();
    if (!this.disposed) {
      this.renderer.setAnimationLoop(this.tick);
    }
    return blob;
  }

  private disposeMesh(mesh: SplatMesh): void {
    (mesh as THREE.Object3D & { dispose?: () => void }).dispose?.();
  }

  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this.onPointerUp);
    this.controls.dispose();
    this.viewHelper.dispose();
    if (this.mesh) {
      this.splatRoot.remove(this.mesh);
      this.disposeMesh(this.mesh);
      this.mesh = null;
    }
    this.grid.geometry.dispose();
    (this.grid.material as THREE.Material).dispose();
    this.axes.geometry.dispose();
    (this.axes.material as THREE.Material).dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
