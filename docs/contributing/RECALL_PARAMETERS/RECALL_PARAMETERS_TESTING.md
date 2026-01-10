# Recall Parameters Feature - Testing & Troubleshooting Guide

## ✅ Checklist Before Testing

### Backend Setup
- [ ] **Stop the current backend** if it's running
- [ ] **Restart the backend** from the InvokeAI root directory:
  ```bash
  cd /home/lstein/Projects/InvokeAI
  # Start the backend (adjust command based on your setup)
  invokeai --web
  # or if using Python directly:
  /home/lstein/invokeai-main/.venv/bin/python -m invokeai.app
  ```
- [ ] Verify backend is running at `http://localhost:9090` (or your configured host:port)

### Frontend Setup
- [ ] **Rebuild the frontend**:
  ```bash
  cd /home/lstein/Projects/InvokeAI/invokeai/frontend/web
  pnpm build
  ```
- [ ] **Clear browser cache** (Ctrl+Shift+Delete or Cmd+Shift+Delete)
- [ ] **Hard refresh the page** (Ctrl+F5 or Cmd+Shift+R)

## 🧪 Testing Steps

### 1. Verify Backend Event Emission
Open a terminal and watch the backend logs:
```bash
# In one terminal, start the backend and keep logs visible
```

Then send a POST request in another terminal:
```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "a beautiful sunset",
    "steps": 25
  }' | jq .
```

**Expected backend log output:**
```
INFO: Emitting recall_parameters_updated event for queue default with 2 parameters
INFO: Successfully emitted recall_parameters_updated event
```

✅ If you see these messages, the backend is working correctly.
❌ If you don't see them, check that:
- The backend was restarted after code changes
- The router is properly imported in api_app.py
- No exceptions are being thrown

### 2. Verify Frontend Event Reception
Open the browser developer console (F12) and go to the **Console** tab.

Make sure you see these messages during the connection:
- `Connected` (debug level)
- Event listeners are being set up

Then send the same POST request again and look for:
```
*** RECALL_PARAMETERS_UPDATED EVENT RECEIVED ***
Recall parameters updated
Applied 2 recall parameters to store
```

✅ If you see these messages, the event is being received.
❌ If you don't see them:
- Check if the frontend is connected to the WebSocket (look for "Connected" message)
- Make sure you're using the newly built frontend (check dist timestamp)
- Try clearing browser cache and hard-refreshing

### 3. Verify UI Updates
After receiving the event, check if the UI fields updated:
- [ ] Positive prompt field updated
- [ ] Negative prompt field updated
- [ ] Steps field updated (if included in POST)
- [ ] CFG Scale field updated (if included in POST)
- [ ] Width field updated (if included in POST)
- [ ] Height field updated (if included in POST)
- [ ] Seed field updated (if included in POST)

## 📝 Full Test Example

### Setup
```bash
# Terminal 1: Start backend with verbose logging
cd /home/lstein/Projects/InvokeAI
/home/lstein/invokeai-main/.venv/bin/python -m invokeai.app 2>&1 | tee backend.log

# Terminal 2: Rebuild frontend
cd /home/lstein/Projects/InvokeAI/invokeai/frontend/web
pnpm build

# Terminal 3: Send test requests
```

### Test Request
```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "a beautiful sunset over mountains",
    "negative_prompt": "blurry, dark",
    "steps": 30,
    "cfg_scale": 8.0,
    "seed": 42,
    "width": 768,
    "height": 512
  }' | jq .
```

### Check Results
1. Look at `backend.log` for emission messages
2. Open browser DevTools Console (F12)
3. Verify all console messages appear
4. Check that all UI fields updated

## 🐛 Troubleshooting

### Issue: No backend logs about emission
**Solution:**
- Restart the backend after code changes
- Check that `emit_recall_parameters_updated` was added to `events_base.py`
- Verify `RecallParametersUpdatedEvent` is in sockets.py QUEUE_EVENTS

### Issue: No frontend console messages
**Solution:**
- Hard refresh the page (Ctrl+F5)
- Check browser console for WebSocket connection errors
- Verify the frontend is using the newly built dist files
- Check that `recall_parameters_updated` listener is in `setEventListeners.tsx`

### Issue: UI fields don't update
**Solution:**
- Check that Redux actions are being dispatched (look for other Redux messages in console)
- Verify the parameter types match (string for prompts, number for numeric values)
- Check browser DevTools for any React errors

### Issue: Connection errors
**Solution:**
- Verify backend is actually running: `curl http://localhost:9090/api/docs`
- Check that CORS is properly configured
- Look for WebSocket connection errors in browser DevTools Network tab

## 📋 Files Changed
The following files were modified to implement this feature:

### Backend
- `invokeai/app/api/routers/recall_parameters.py` - New router for the API
- `invokeai/app/api_app.py` - Registers the router
- `invokeai/app/services/events/events_common.py` - New `RecallParametersUpdatedEvent`
- `invokeai/app/services/events/events_base.py` - New `emit_recall_parameters_updated` method
- `invokeai/app/api/sockets.py` - Registers the event in QUEUE_EVENTS

### Frontend
- `invokeai/frontend/web/src/services/events/types.ts` - Added event type
- `invokeai/frontend/web/src/services/events/setEventListeners.tsx` - Added listener and handler

## 🔄 API Reference

### Endpoint
```
POST /api/v1/recall/{queue_id}
```

### Parameters
```json
{
  "positive_prompt": "string",
  "negative_prompt": "string",
  "steps": "number",
  "cfg_scale": "number",
  "seed": "number",
  "width": "number",
  "height": "number",
  "refiner_steps": "number",
  "refiner_cfg_scale": "number",
  "clip_skip": "number",
  "seamless_x": "boolean",
  "seamless_y": "boolean"
}
```

All parameters are optional. Only send the parameters you want to update.

### Example Request
```bash
curl -X POST http://localhost:9090/api/v1/recall/default \
  -H "Content-Type: application/json" \
  -d '{"positive_prompt": "test", "steps": 20}'
```

### Example Response
```json
{
  "status": "success",
  "queue_id": "default",
  "updated_count": 2,
  "parameters": {
    "positive_prompt": "test",
    "steps": 20
  }
}
```
