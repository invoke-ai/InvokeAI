type Props = {
  /**
   * The amplitude of the wave. 0 is a straight line, higher values create more pronounced waves.
   */
  amplitude: number;
  /**
   * The number of segments in the line. More segments create a smoother wave.
   */
  segments?: number;
  /**
   * The color of the wave.
   */
  stroke: string;
  /**
   * The width of the wave.
   */
  strokeWidth: number;
  /**
   * The width of the SVG.
   */
  width: number;
  /**
   * The height of the SVG.
   */
  height: number;
};

const WavyLine = ({ amplitude, stroke, strokeWidth, width, height, segments = 5 }: Props) => {
  // Calculate the path dynamically based on waviness
  const generatePath = () => {
    if (amplitude === 0) {
      // If waviness is 0, return a straight line
      return `M0,${height / 2} L${width},${height / 2}`;
    }

    const clampedAmplitude = Math.min(height / 2, amplitude); // Cap amplitude to half the height
    const segmentWidth = width / segments;
    let path = `M0,${height / 2}`; // Start in the middle of the left edge

    // Loop through each segment and alternate the y position to create waves
    for (let i = 1; i <= segments; i++) {
      const x = i * segmentWidth;
      const y = height / 2 + (i % 2 === 0 ? clampedAmplitude : -clampedAmplitude);
      path += ` Q${x - segmentWidth / 2},${y} ${x},${height / 2}`;
    }

    return path;
  };

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} xmlns="http://www.w3.org/2000/svg">
      <path d={generatePath()} fill="none" stroke={stroke} strokeWidth={strokeWidth} />
    </svg>
  );
};

export default WavyLine;
