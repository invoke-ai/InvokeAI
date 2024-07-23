these ideas are trying to figure out the macOS photos UX where you use scroll position instead of page number to go to a specific range of images.

### Brute Force

Two new methods/endpoints:

- `get_image_names`: gets a list of ordered image names for the query params (e.g. board, starred_first)
- `get_images_by_name`: gets the dtos for a list of image names

Broad strokes of client handling:

- Fetch a list of all image names for a board.
- Render a big scroll area, large enough to hold all images. The list of image names is passed to `react-virtuoso` (virtualized list lib).
- As you scroll, we use the rangeChanged callback from `react-virtuoso`, which provides the indices of the currently-visible images in the list of all images. These indices map back to the list of image names from which we can derive the list of image names we need to fetch
- Debounce the rnageChanged callback
- Call the `get_images_by_name` endpoint with hte image names to fetch, use the result to update the `getImageDTO` query cache. De-duplicate the image_names against existing cache before fetching so we aren't requesting the smae data over and over
- Each item/image in the virtualized list fetches its image DTO from the cache _without initiating a network request_. it just kinda waits until the image is in the cache and then displays it

this is roughed out in this branch

#### FATAL FLAW

Once you generate an image, you want to do an optimistic update and insert its name into the big ol' image_names list right? well, where do you insert it? depends on the query parms that can affect the sort order and which images are shown... we only have the image names at this point so we can't easily figure out where to insert

workarounds (?):

- along with the image names, we retrieve `starred_first` and `created_at`. then from the query params we can easily figure out where to insert the new image into the list to match the sort that he backend will be doing. eh
- fetch `starred_first` images separately? so we don't have to worry about inserting the image into the right spot?

ahh but also metadata search... we won't know when to insert the image into the list if the user has a search term...

#### Sub-idea

Ok let's still use pagination but use virtuoso to tell us which page we are on.

virtuoso has an alternate mode where you just tell it how many items you have and it renders each item, passing only an index to it. Maybe we can derive the limit and offset from this information. here's an untested idea:

- pass virtuoso the board count
- Instead of rendering individual images in the list, we render pages (ranges) of images. The list library’s rangeChanged indices now refer to pages or ranges. To the user, it still looks like a bunch of individual images, but internally we group it into pages/ranges of whatever size.
- The page/range size is calculated via DOM, or we can rely on virtuoso to tell us how many items are to be rendered. only thing is it the number can different depending on scroll position, so we'd probably want to like take `endIndex - startIndex` as the limit, add 20% buffer to each end of the limit and round it to the nearest multiple of 5 or 10. that would give us a consistent limit
- then we can derive offset from that value

still has the issue where we aren't sure if we should trigger a image list cache invalidation...

### More Efficient Pagination

sql OFFSET requires a scan thru the whole table upt othe offset. that means the higher the offset, the slower the query. unsure of the practical impact of this, probably negligible for us right now.

I did some quick experiments with cursor/keyset pagination, using an image name as the cursor. this doesn't have the perf issue w/ offset.

Also! This kind of pagination is unaffected by insertions and deletions, which is a problem for limit/offset pagination. When you insert or delete an image, it doesn't shift images at higher pages down. I think this pagination strategy suits our gallery better than limit/offset, given how volatile it is with adding and removing images regularly.

see the `test_keyset` notebook for implementation (also some scattered methods in services as I was fiddling withh it)

may be some way to use this pagination strat in combination with the above ideas to more elegantly handle inserting and deleting images...

### Alternative approach to the whole "how do we know when to insert new images in the list (or invalidate the list cache)" issue

What if we _always_ invalidate the cache when youa re at the top of the list ,but never invalidate it when you have scrolled down?
