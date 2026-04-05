import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../tests/test-utils';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from './IAIImageFallback';

describe('IAINoContentFallback', () => {
  it('renders without a label', () => {
    const { container } = renderWithProviders(<IAINoContentFallback />);
    expect(container.firstChild).toBeInTheDocument();
  });

  it('renders the label text when provided', () => {
    renderWithProviders(<IAINoContentFallback label="No images found" />);
    expect(screen.getByText('No images found')).toBeInTheDocument();
  });

  it('does not render a label element when label is not provided', () => {
    renderWithProviders(<IAINoContentFallback />);
    expect(screen.queryByRole('paragraph')).not.toBeInTheDocument();
  });

  it('does not render an icon when icon is null', () => {
    const { container } = renderWithProviders(<IAINoContentFallback icon={null} />);
    expect(container.querySelector('svg')).not.toBeInTheDocument();
  });

  it('renders an icon by default', () => {
    const { container } = renderWithProviders(<IAINoContentFallback />);
    expect(container.querySelector('svg')).toBeInTheDocument();
  });
});

describe('IAINoContentFallbackWithSpinner', () => {
  it('renders without a label', () => {
    const { container } = renderWithProviders(<IAINoContentFallbackWithSpinner />);
    expect(container.firstChild).toBeInTheDocument();
  });

  it('renders the label text when provided', () => {
    renderWithProviders(<IAINoContentFallbackWithSpinner label="Loading images..." />);
    expect(screen.getByText('Loading images...')).toBeInTheDocument();
  });

  it('does not render a label when label is not provided', () => {
    renderWithProviders(<IAINoContentFallbackWithSpinner />);
    expect(screen.queryByText('Loading images...')).not.toBeInTheDocument();
  });
});
