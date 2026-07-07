import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper.js';

import { remapPointerForViewHelper } from './viewHelperPointer';

// Cap the drawing buffer's largest dimension so deep stage zooms don't allocate absurd buffers.
const MAX_DRAWING_BUFFER_DIM = 4096;

// Object-rotation speed for Alt+drag, radians per screen px (~0.46°/px: a 200px drag is ~a quarter turn).
const OBJECT_ROTATE_SPEED = 0.008;

/**
 * A transparent three.js + Spark viewport for a single 3D Gaussian splat, designed to sit directly on the
 * canvas: orbit/zoom/pan camera and a clickable corner navigation gizmo (ViewHelper, shown only while the
 * pointer is over the viewport). Alt+drag rotates the object itself (all three axes) — OrbitControls keeps
 * the horizon level, so poses like a diagonal lean are unreachable by camera orbit alone. The background is
 * always transparent so the canvas shows through — the live view is the compositing preview. Capture
 * renders the object alone at a given pixel size.
 *
 * The element hosting this scene is CSS-scaled by the canvas stage transform, so the drawing buffer is
 * sized at (layout px × devicePixelRatio × stage scale) to stay crisp at any zoom — see setStageScale.
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
  private readonly spark: SparkRenderer;
  private readonly viewHelper: ViewHelper;
  private readonly clock: THREE.Clock;
  private readonly splatRoot: THREE.Group;
  private readonly resizeObserver: ResizeObserver;
  private readonly onPointerDown: (e: PointerEvent) => void;
  private readonly onPointerUp: (e: PointerEvent) => void;
  private readonly onObjectRotateStart: (e: PointerEvent) => void;
  private readonly onObjectRotateMove: (e: PointerEvent) => void;
  private readonly onObjectRotateEnd: (e: PointerEvent) => void;
  private pointerDownPos: { x: number; y: number } | null = null;
  private objectRotate: { pointerId: number; lastX: number; lastY: number } | null = null;
  private mesh: SplatMesh | null = null;
  private stageScale = 1;
  private gizmoVisible = false;
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

    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);

    this.splatRoot = new THREE.Group();
    this.splatRoot.rotation.x = Math.PI;
    this.scene.add(this.splatRoot);

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
      this.pointerDownPos = null;
      if (down && this.gizmoVisible && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
        // ViewHelper's own hit test breaks while the stage transform CSS-scales this viewport (it mixes
        // visual and layout coordinates) — remap so axis clicks land at any stage zoom.
        const el = this.renderer.domElement;
        const remapped = remapPointerForViewHelper(
          { rect: el.getBoundingClientRect(), offsetWidth: el.offsetWidth, offsetHeight: el.offsetHeight },
          e.clientX,
          e.clientY
        );
        this.viewHelper.handleClick(remapped as PointerEvent);
      }
    };
    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);

    // Alt+drag rotates the object itself about its center, on the camera's screen axes. Registered on the
    // container in the capture phase so it claims the gesture before OrbitControls' own pointerdown runs.
    this.onObjectRotateStart = (e) => {
      if (!e.altKey || e.button !== 0 || this.disposed || this.objectRotate) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      this.objectRotate = { pointerId: e.pointerId, lastX: e.clientX, lastY: e.clientY };
      window.addEventListener('pointermove', this.onObjectRotateMove);
      window.addEventListener('pointerup', this.onObjectRotateEnd);
      window.addEventListener('pointercancel', this.onObjectRotateEnd);
    };
    this.onObjectRotateMove = (e) => {
      const rot = this.objectRotate;
      if (!rot || e.pointerId !== rot.pointerId) {
        return;
      }
      const dx = e.clientX - rot.lastX;
      const dy = e.clientY - rot.lastY;
      rot.lastX = e.clientX;
      rot.lastY = e.clientY;
      // Rotate about the camera's current screen axes so the object follows the pointer from any view
      // angle; successive two-axis increments compose to reach any orientation, including roll.
      const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
      const q = new THREE.Quaternion();
      this.splatRoot.quaternion.premultiply(q.setFromAxisAngle(up, dx * OBJECT_ROTATE_SPEED));
      this.splatRoot.quaternion.premultiply(q.setFromAxisAngle(right, dy * OBJECT_ROTATE_SPEED));
    };
    this.onObjectRotateEnd = (e) => {
      if (!this.objectRotate || e.pointerId !== this.objectRotate.pointerId) {
        return;
      }
      this.objectRotate = null;
      window.removeEventListener('pointermove', this.onObjectRotateMove);
      window.removeEventListener('pointerup', this.onObjectRotateEnd);
      window.removeEventListener('pointercancel', this.onObjectRotateEnd);
    };
    container.addEventListener('pointerdown', this.onObjectRotateStart, true);

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
    this.renderer.setClearColor(0x000000, 0); // transparent — the canvas beneath is the background
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
    if (this.gizmoVisible) {
      this.viewHelper.render(this.renderer);
    }
  };

  /**
   * The canvas stage scale currently applied (via CSS transform) to this viewport's container. Folded into
   * the renderer's pixel ratio so the drawing buffer matches on-screen pixels — crisp at any zoom.
   */
  setStageScale = (scale: number): void => {
    const next = Math.max(0.001, scale || 1);
    if (next === this.stageScale) {
      return;
    }
    this.stageScale = next;
    this.resize();
  };

  /** Show/hide the corner navigation gizmo (shown only while the pointer is over the viewport). */
  setGizmoVisible = (visible: boolean): void => {
    this.gizmoVisible = visible;
  };

  private effectivePixelRatio(w: number, h: number): number {
    const dpr = window.devicePixelRatio || 1;
    // Quantize so wheel-zooming the stage doesn't reallocate the drawing buffer on every tick.
    const quantized = Math.max(0.25, Math.round(dpr * this.stageScale * 4) / 4);
    return Math.min(quantized, MAX_DRAWING_BUFFER_DIM / Math.max(w, h, 1));
  }

  resize = (): void => {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setPixelRatio(this.effectivePixelRatio(w, h));
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
   * Capture the object alone (no gizmo, transparent background) at the given pixel size, from the user's
   * current camera (WYSIWYG). Renders at pixelRatio 1 so the blob's intrinsic size equals width×height. The
   * live render loop is paused during capture to avoid it overwriting the frame before toBlob reads it.
   */
  async capture(width: number, height: number): Promise<Blob | null> {
    this.renderer.setAnimationLoop(null);
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
    // Restore the live preview (resize() also restores the stage-scale-aware pixel ratio).
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
    this.container.removeEventListener('pointerdown', this.onObjectRotateStart, true);
    window.removeEventListener('pointermove', this.onObjectRotateMove);
    window.removeEventListener('pointerup', this.onObjectRotateEnd);
    window.removeEventListener('pointercancel', this.onObjectRotateEnd);
    this.controls.dispose();
    this.viewHelper.dispose();
    if (this.mesh) {
      this.splatRoot.remove(this.mesh);
      this.disposeMesh(this.mesh);
      this.mesh = null;
    }
    // SparkRenderer owns sort/LOD web workers and GPU-side state that only its dispose() releases.
    this.scene.remove(this.spark);
    this.spark.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
