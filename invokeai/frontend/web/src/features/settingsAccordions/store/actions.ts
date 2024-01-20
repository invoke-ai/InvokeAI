import { createAction } from '@reduxjs/toolkit';

export const expanderToggled = createAction<{ id: string; isOpen: boolean }>(
  'parameters/expanderToggled'
);

export const standaloneAccordionToggled = createAction<{
  id: string;
  isOpen: boolean;
}>('parameters/standaloneAccordionToggled');
