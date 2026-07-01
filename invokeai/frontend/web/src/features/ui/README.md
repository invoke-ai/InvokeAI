# UI/Layout

We use https://github.com/mathuo/dockview for layout. This library supports resizable and dockable panels. Users can drag and drop panels to rearrange them.

The intention when adopting this library was to allow users to create their own custom layouts and save them. However, this feature is not yet implemented and each tab only has a predefined layout.

This works well, but it _is_ fairly complex. You can see that we've needed to write a fairly involved API to manage the layouts: invokeai/frontend/web/src/features/ui/layouts/navigation-api.ts

And the layouts themselves are awkward to define, especially when compared to plain JSX: invokeai/frontend/web/src/features/ui/layouts/generate-tab-auto-layout.tsx

This complexity may or may not be worth it.

## Previous approach

Previously we used https://github.com/bvaughn/react-resizable-panels and simple JSX components.

This library is great except it doesn't support absolute size constraints, only relative/percentage constraints. We had a brittle abstraction layer on top of it to try to enforce minimum pixel sizes for panels but it was janky and had FP precision issues causing drifting sizes.

It also doesn't support dockable panels.

## Future possibilities

1. Continue with dockview and implement custom layout saving/loading. We experimented with this and it was _really_ nice. We defined a component for each panel type and use react context to manage state. But we thought that it would be confusing for most users, so we flagged it for a future iteration and instead shipped with predefined layouts.
2. Switch to a simpler layout library or roll our own.

In hindsight, we should have skipped dockview and found something else that was simpler until we were ready to invest in custom layouts.
