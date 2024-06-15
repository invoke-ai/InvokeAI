import type { ActionReducerMapBuilder, PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

type MySlice = {
  flavour: 'vanilla' | 'chocolate' | 'strawberry';
  sprinkles: boolean;
  customers: { id: string; name: string }[];
};
const initialStateMySlice: MySlice = { flavour: 'vanilla', sprinkles: false, customers: [] };

const reducersInAnotherFile: SliceCaseReducers<MySlice> = {
  sprinklesToggled: (state) => {
    state.sprinkles = !state.sprinkles;
  },
  customerAdded: {
    reducer: (state, action: PayloadAction<{ id: string; name: string }>) => {
      state.customers.push(action.payload);
    },
    prepare: (name: string) => ({ payload: { name, id: crypto.randomUUID() } }),
  },
};

const extraReducersInAnotherFile = (builder: ActionReducerMapBuilder<MySlice>) => {
  builder.addCase(otherSlice.actions.fooChanged, (state, action) => {
    if (action.payload === 'bar') {
      state.flavour = 'vanilla';
    }
  });
};

export const mySlice = createSlice({
  name: 'mySlice',
  initialState: initialStateMySlice,
  reducers: {
    ...reducersInAnotherFile,
    flavourChanged: (state, action: PayloadAction<MySlice['flavour']>) => {
      state.flavour = action.payload;
    },
  },
  extraReducers: extraReducersInAnotherFile,
});

type OtherSlice = {
  something: string;
};

const initialStateOtherSlice: OtherSlice = { something: 'foo' };

export const otherSlice = createSlice({
  name: 'otherSlice',
  initialState: initialStateOtherSlice,
  reducers: {
    fooChanged: (state, action: PayloadAction<string>) => {
      state.something = action.payload;
    },
  },
});
