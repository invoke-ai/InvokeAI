const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
const LEGACY_WAVE_AMPLITUDE_SCALE = 10;

export const getDenoisingStrengthWavePath = (
  strength: number,
  width: number,
  height: number,
  segments: number
): string => {
  const centerY = height / 2;
  const amplitude = Math.min(centerY, clamp01(strength) * LEGACY_WAVE_AMPLITUDE_SCALE);
  if (amplitude === 0) {
    return `M0,${centerY} L${width},${centerY}`;
  }

  const segmentWidth = width / segments;
  let path = `M0,${centerY}`;
  for (let index = 1; index <= segments; index += 1) {
    const x = index * segmentWidth;
    const controlX = x - segmentWidth / 2;
    const controlY = centerY + (index % 2 === 0 ? amplitude : -amplitude);
    path += ` Q${controlX},${controlY} ${x},${centerY}`;
  }
  return path;
};

const WAVE_WIDTH = 100;
const WAVE_HEIGHT = 14;
const WAVE_SEGMENTS = 5;

/** A value-reactive replacement for the denoising slider's straight track. */
export const DenoisingStrengthWave = ({ value }: { value: number }) => {
  const path = getDenoisingStrengthWavePath(value, WAVE_WIDTH, WAVE_HEIGHT, WAVE_SEGMENTS);

  return (
    <svg
      aria-hidden
      height={WAVE_HEIGHT}
      preserveAspectRatio="none"
      style={{ display: 'block', flexShrink: 0, marginLeft: 8, pointerEvents: 'none' }}
      viewBox={`0 0 ${WAVE_WIDTH} ${WAVE_HEIGHT}`}
      width={56}
    >
      <path
        d={path}
        fill="none"
        stroke="var(--chakra-colors-accent-solid)"
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};
