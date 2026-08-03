// Published as the deep specifier `@platform/ui/hints`, deliberately absent from
// the `@platform/ui` barrel: that barrel sits near its `maxDirectImporters`
// budget, and every hint call site would otherwise count against it.
export * from './FeatureHint';
export * from './hintRegistry';
export * from './hintsContext';
