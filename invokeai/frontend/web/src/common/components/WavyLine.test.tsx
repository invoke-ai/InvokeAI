import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import WavyLine from './WavyLine';

describe('WavyLine', () => {
  it('renders an SVG element', () => {
    const { container } = render(<WavyLine amplitude={5} stroke="red" strokeWidth={2} width={100} height={40} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('sets the correct SVG dimensions', () => {
    const { container } = render(<WavyLine amplitude={5} stroke="blue" strokeWidth={1} width={200} height={50} />);
    const svg = container.querySelector('svg');
    expect(svg).toHaveAttribute('width', '200');
    expect(svg).toHaveAttribute('height', '50');
  });

  it('renders a straight line when amplitude is 0', () => {
    const { container } = render(<WavyLine amplitude={0} stroke="green" strokeWidth={2} width={100} height={40} />);
    const path = container.querySelector('path');
    expect(path).toHaveAttribute('d', 'M0,20 L100,20');
  });

  it('renders a wavy path when amplitude is non-zero', () => {
    const { container } = render(<WavyLine amplitude={10} stroke="green" strokeWidth={2} width={100} height={40} />);
    const path = container.querySelector('path');
    const d = path?.getAttribute('d') ?? '';
    expect(d).toContain('Q');
  });

  it('applies the correct stroke color to the path', () => {
    const { container } = render(<WavyLine amplitude={5} stroke="#ff0000" strokeWidth={2} width={100} height={40} />);
    const path = container.querySelector('path');
    expect(path).toHaveAttribute('stroke', '#ff0000');
  });

  it('applies the correct stroke width to the path', () => {
    const { container } = render(<WavyLine amplitude={5} stroke="red" strokeWidth={3} width={100} height={40} />);
    const path = container.querySelector('path');
    expect(path).toHaveAttribute('stroke-width', '3');
  });
});
