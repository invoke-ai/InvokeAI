const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

export const getDenoisingStrengthWavePath = (
  strength: number,
  width: number,
  height: number,
  segments: number
): string => {
  const centerY = height / 2;
  const amplitude = clamp01(strength) * Math.max(0, centerY - 1);
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
const WAVE_HEIGHT = 12;
const WAVE_SEGMENTS = 10;

/** A value-reactive replacement for the denoising slider's straight track. */
export const DenoisingStrengthWave = ({ value }: { value: number }) => {
  const clampedValue = clamp01(value);
  const path = getDenoisingStrengthWavePath(clampedValue, WAVE_WIDTH, WAVE_HEIGHT, WAVE_SEGMENTS);

  return (
    <svg
      aria-hidden
      height={WAVE_HEIGHT}
      preserveAspectRatio="none"
      style={{ left: 0, pointerEvents: 'none', position: 'absolute', top: '50%', transform: 'translateY(-50%)' }}
      viewBox={`0 0 ${WAVE_WIDTH} ${WAVE_HEIGHT}`}
      width="100%"
    >
      <path
        d={path}
        fill="none"
        opacity="0.55"
        stroke="var(--chakra-colors-fg-muted)"
        strokeWidth="1.25"
        vectorEffect="non-scaling-stroke"
      />
      <path
        d={path}
        fill="none"
        pathLength={100}
        stroke="var(--chakra-colors-accent-solid)"
        strokeDasharray={`${clampedValue * 100} 100`}
        strokeWidth="1.75"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};
