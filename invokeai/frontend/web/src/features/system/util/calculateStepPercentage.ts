export const calculateStepPercentage = (step: number, total_steps: number, order: number) => {
  if (total_steps === 0) {
    return 0;
  }

  // we add one extra to step so that the progress bar will be full when denoise completes

  if (order === 2) {
    return Math.floor((step + 1 + 1) / 2) / Math.floor((total_steps + 1) / 2);
  }

  return (step + 1 + 1) / (total_steps + 1);
};
