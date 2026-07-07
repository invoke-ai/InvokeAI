import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper.js';

import { remapPointerForViewHelper, VIEW_HELPER_DIM } from './viewHelperPointer';

// Cap the drawing buffer's largest dimension so deep stage zooms don't allocate absurd buffers.
const MAX_DRAWING_BUFFER_DIM = 4096;

// Object-rotation speed in rotate-object mode, radians per screen px (~0.46°/px: 200px ≈ a quarter turn).
const OBJECT_ROTATE_SPEED = 0.008;

// Pivot marker: shown while a gesture is active, at the point the gesture rotates around.
const PIVOT_MARKER_COLOR = 0x66b2ff; // ≈ invokeBlue.300, matches the overlay chrome
const PIVOT_MARKER_SCREEN_SCALE = 0.008; // world size per unit of camera distance ≈ constant screen size
const PIVOT_FADE_RATE = 8; // opacity lerp rate per second (~0.3s fade)
const WORLD_ORIGIN = new THREE.Vector3(0, 0, 0);

const buildPivotMarker = (): {
  group: THREE.Group;
  materials: (THREE.MeshBasicMaterial | THREE.LineBasicMaterial)[];
} => {
  const dotMaterial = new THREE.MeshBasicMaterial({
    color: PIVOT_MARKER_COLOR,
    transparent: true,
    opacity: 0,
    depthTest: false,
    depthWrite: false,
  });
  const lineMaterial = new THREE.LineBasicMaterial({
    color: PIVOT_MARKER_COLOR,
    transparent: true,
    opacity: 0,
    depthTest: false,
    depthWrite: false,
  });
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.6, 12, 8), dotMaterial);
  const cross = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-2.4, 0, 0),
      new THREE.Vector3(2.4, 0, 0),
      new THREE.Vector3(0, -2.4, 0),
      new THREE.Vector3(0, 2.4, 0),
      new THREE.Vector3(0, 0, -2.4),
      new THREE.Vector3(0, 0, 2.4),
    ]),
    lineMaterial
  );
  // Drawn on top (no depth test, late render order): the pivot usually sits inside the splat body,
  // which would otherwise occlude it.
  dot.renderOrder = 999;
  cross.renderOrder = 999;
  const group = new THREE.Group();
  group.add(dot, cross);
  group.visible = false;
  return { group, materials: [dotMaterial, lineMaterial] };
};

/**
 * A transparent three.js + Spark viewport for a single 3D Gaussian splat, designed to sit directly on the
 * canvas: orbit/zoom/pan camera and a clickable corner navigation gizmo (ViewHelper, shown only while the
 * pointer is over the viewport). A toggleable rotate-object mode (setRotateObjectMode) makes left-drag
 * rotate the object itself (all three axes) instead of orbiting — OrbitControls keeps the horizon level,
 * so poses like a diagonal lean are unreachable by camera orbit alone. While any gesture is active, a
 * crosshair marker fades in at the point the gesture rotates around (the orbit target, or the object
 * origin in rotate-object mode); it is never rendered into captures. The background is always transparent
 * so the canvas shows through — the live view is the compositing preview. Capture renders the object alone
 * at a given pixel size.
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
  private readonly onContextMenu: (e: MouseEvent) => void;
  private pointerDownPos: { x: number; y: number } | null = null;
  private objectRotate: { pointerId: number; lastX: number; lastY: number } | null = null;
  private rotateObjectMode = false;
  private readonly pivotMarker: THREE.Group;
  private readonly pivotMaterials: (THREE.MeshBasicMaterial | THREE.LineBasicMaterial)[];
  private pivotOpacity = 0;
  private pivotTargetOpacity = 0;
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

    // Pivot marker: fades in while the user orbits/pans/zooms (or rotates the object) so the rotation
    // center is visible; tick() keeps it positioned, screen-sized, and faded.
    const pivot = buildPivotMarker();
    this.pivotMarker = pivot.group;
    this.pivotMaterials = pivot.materials;
    this.scene.add(this.pivotMarker);
    this.controls.addEventListener('start', this.showPivotMarker);
    this.controls.addEventListener('end', this.hidePivotMarker);

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

    // In rotate-object mode, left-drag rotates the object itself about its center on the camera's screen
    // axes. Registered on the container in the capture phase so it claims the gesture before OrbitControls'
    // own pointerdown runs; wheel-zoom and right-drag pan still reach OrbitControls, and clicks on the
    // corner gizmo pass through so view snapping keeps working.
    this.onObjectRotateStart = (e) => {
      if (!this.rotateObjectMode || e.button !== 0 || this.disposed || this.objectRotate || this.isOverGizmo(e)) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      this.objectRotate = { pointerId: e.pointerId, lastX: e.clientX, lastY: e.clientY };
      this.showPivotMarker();
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
      this.hidePivotMarker();
      window.removeEventListener('pointermove', this.onObjectRotateMove);
      window.removeEventListener('pointerup', this.onObjectRotateEnd);
      window.removeEventListener('pointercancel', this.onObjectRotateEnd);
    };
    container.addEventListener('pointerdown', this.onObjectRotateStart, true);

    // Right-drag pans the camera, and contextmenu fires on right-button RELEASE — without this, every pan
    // ends by popping the canvas context menu. preventDefault kills the browser menu; stopPropagation keeps
    // the event from bubbling to the app's canvas context-menu trigger.
    this.onContextMenu = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };
    container.addEventListener('contextmenu', this.onContextMenu);

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
    // Pivot marker: track the active gesture's rotation center at a constant screen size, fading around
    // interactions.
    this.pivotOpacity += (this.pivotTargetOpacity - this.pivotOpacity) * Math.min(1, delta * PIVOT_FADE_RATE);
    if (this.pivotTargetOpacity === 0 && this.pivotOpacity < 0.02) {
      this.pivotOpacity = 0;
    }
    this.pivotMarker.visible = this.pivotOpacity > 0;
    if (this.pivotMarker.visible) {
      // Orbit/pan/zoom rotate around controls.target; rotate-object mode spins the object about its origin.
      const pivot = this.objectRotate ? WORLD_ORIGIN : this.controls.target;
      this.pivotMarker.position.copy(pivot);
      this.pivotMarker.scale.setScalar(
        Math.max(1e-6, this.camera.position.distanceTo(pivot) * PIVOT_MARKER_SCREEN_SCALE)
      );
      for (const material of this.pivotMaterials) {
        material.opacity = this.pivotOpacity * 0.9;
      }
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

  /** When enabled, left-drag rotates the object itself (any axis) instead of orbiting the camera. */
  setRotateObjectMode = (enabled: boolean): void => {
    this.rotateObjectMode = enabled;
  };

  private showPivotMarker = (): void => {
    this.pivotTargetOpacity = 1;
  };

  private hidePivotMarker = (): void => {
    this.pivotTargetOpacity = 0;
  };

  /** Whether the pointer is over the ViewHelper's corner viewport (compared in the element's layout space). */
  private isOverGizmo(e: PointerEvent): boolean {
    if (!this.gizmoVisible) {
      return false;
    }
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    const scaleX = rect.width / (el.offsetWidth || 1) || 1;
    const scaleY = rect.height / (el.offsetHeight || 1) || 1;
    const layoutX = (e.clientX - rect.left) / scaleX;
    const layoutY = (e.clientY - rect.top) / scaleY;
    return layoutX >= el.offsetWidth - VIEW_HELPER_DIM && layoutY >= el.offsetHeight - VIEW_HELPER_DIM;
  }

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
    this.pivotMarker.visible = false; // never bake the pivot marker into the layer; tick() restores it
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
    this.container.removeEventListener('contextmenu', this.onContextMenu);
    window.removeEventListener('pointermove', this.onObjectRotateMove);
    window.removeEventListener('pointerup', this.onObjectRotateEnd);
    window.removeEventListener('pointercancel', this.onObjectRotateEnd);
    this.controls.removeEventListener('start', this.showPivotMarker);
    this.controls.removeEventListener('end', this.hidePivotMarker);
    this.scene.remove(this.pivotMarker);
    this.pivotMarker.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
      }
    });
    for (const material of this.pivotMaterials) {
      material.dispose();
    }
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
