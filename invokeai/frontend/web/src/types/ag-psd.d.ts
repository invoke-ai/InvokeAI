declare module 'ag-psd' {
  export interface PsdLayer {
    name: string;
    left: number;
    top: number;
    right: number;
    bottom: number;
    opacity: number;
    hidden: boolean;
    blendMode: 'normal' | 'multiply' | 'screen' | 'overlay' | 'darken' | 'lighten' | 'colorDodge' | 'colorBurn' | 'hardLight' | 'softLight' | 'difference' | 'exclusion' | 'hue' | 'saturation' | 'color' | 'luminosity';
    canvas: HTMLCanvasElement;
  }

  export interface PsdDocument {
    width: number;
    height: number;
    channels: number;
    bitsPerChannel: number;
    colorMode: number;
    children: PsdLayer[];
  }

  export function writePsd(psd: PsdDocument): ArrayBuffer;
} 