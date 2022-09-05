// TODO: use Enums?

// Valid samplers
export const SAMPLERS: Array<string> = [
    'ddim',
    'plms',
    'k_lms',
    'k_dpm_2',
    'k_dpm_2_a',
    'k_euler',
    'k_euler_a',
    'k_heun',
];

// Valid image widths
export const WIDTHS: Array<number> = [
    64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
    1024,
];

// Valid image heights
export const HEIGHTS: Array<number> = [
    64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
    1024,
];

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<number> = [0, 2, 4];
