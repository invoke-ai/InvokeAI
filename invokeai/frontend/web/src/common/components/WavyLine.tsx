const WavyLine = ({ waviness }: { waviness: number }) => {
  const width = 40; // Width of the SVG
  const height = 14; // Height of the SVG
  const segments = 5; // Number of segments in the line (more segments = smoother wave)

  // Calculate the path dynamically based on waviness
  const generatePath = () => {
    if (waviness === 0) {
      // If waviness is 0, return a straight line
      return `M0,${height / 2} L${width},${height / 2}`;
    }

    const amplitude = Math.min(height / 2, waviness); // Cap amplitude to half the height
    const segmentWidth = width / segments;
    let path = `M0,${height / 2}`; // Start in the middle of the left edge

    // Loop through each segment and alternate the y position to create waves
    for (let i = 1; i <= segments; i++) {
      const x = i * segmentWidth;
      const y = height / 2 + (i % 2 === 0 ? amplitude : -amplitude);
      path += ` Q${x - segmentWidth / 2},${y} ${x},${height / 2}`;
    }

    return path;
  };

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} xmlns="http://www.w3.org/2000/svg">
      <path d={generatePath()} fill="none" stroke="#4299e1" strokeWidth="3" />
    </svg>
  );
};

export default WavyLine;
