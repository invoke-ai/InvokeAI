const socket = io();

function resetForm() {
  var form = document.getElementById('generate-form');
  form.querySelector('fieldset').removeAttribute('disabled');
}

function initProgress(totalSteps, showProgressImages) {
  // TODO: Progress could theoretically come from multiple jobs at the same time (in the future)
  let progressSectionEle = document.querySelector('#progress-section');
  progressSectionEle.style.display = 'initial';
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('max', totalSteps);

  let progressImageEle = document.querySelector('#progress-image');
  progressImageEle.src = BLANK_IMAGE_URL;
  progressImageEle.style.display = showProgressImages ? 'initial': 'none';
}

function setProgress(step, totalSteps, src) {
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('value', step);

  if (src) {
    let progressImageEle = document.querySelector('#progress-image');
    progressImageEle.src = src;
  }
}

function resetProgress() {
  let progressSectionEle = document.querySelector('#progress-section');
  progressSectionEle.style.display = 'none';
  let progressEle = document.querySelector('#progress-bar');
  progressEle.setAttribute('value', 0);
}

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.readAsDataURL(file);
        r.onload = () => resolve(r.result);
        r.onerror = (error) => reject(error);
    });
}

function appendOutput(src, seed, config) {
    let outputNode = document.createElement("figure");
    let altText = seed.toString() + " | " + config.prompt;

    const figureContents = `
        <a href="${src}" target="_blank">
            <img src="${src}" alt="${altText}" title="${altText}">
        </a>
        <figcaption>${seed}</figcaption>
    `;

    outputNode.innerHTML = figureContents;
    let figcaption = outputNode.querySelector('figcaption')

    // Reload image config
    figcaption.addEventListener('click', () => {
        let form = document.querySelector("#generate-form");
        for (const [k, v] of new FormData(form)) {
            if (k == 'initimg') { continue; }
            form.querySelector(`*[name=${k}]`).value = config[k];
        }
        if (config.variation_amount > 0 || config.with_variations != '') {
            document.querySelector("#seed").value = config.seed;
        } else {
            document.querySelector("#seed").value = seed;
        }

        if (config.variation_amount > 0) {
            let oldVarAmt = document.querySelector("#variation_amount").value
            let oldVariations = document.querySelector("#with_variations").value
            let varSep = ''
            document.querySelector("#variation_amount").value = 0;
            if (document.querySelector("#with_variations").value != '') {
                varSep = ","
            }
            document.querySelector("#with_variations").value = oldVariations + varSep + seed + ':' + config.variation_amount
        }

        saveFields(document.querySelector("#generate-form"));
    });

    document.querySelector("#results").prepend(outputNode);
    document.querySelector("#no-results-message")?.remove();
}

function saveFields(form) {
    for (const [k, v] of new FormData(form)) {
        if (typeof v !== 'object') { // Don't save 'file' type
            localStorage.setItem(k, v);
        }
    }
}

function loadFields(form) {
    for (const [k, v] of new FormData(form)) {
        const item = localStorage.getItem(k);
        if (item != null) {
            form.querySelector(`*[name=${k}]`).value = item;
        }
    }
}

function clearFields(form) {
    localStorage.clear();
    let prompt = form.prompt.value;
    form.reset();
    form.prompt.value = prompt;
}

const BLANK_IMAGE_URL = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>';
async function generateSubmit(form) {
    //const prompt = document.querySelector("#prompt").value;

    // Convert file data to base64
    let formData = Object.fromEntries(new FormData(form));
    formData.initimg_name = formData.initimg.name
    formData.initimg = formData.initimg.name !== '' ? await toBase64(formData.initimg) : null;

    let strength = formData.strength;
    let totalSteps = formData.initimg ? Math.floor(strength * formData.steps) : formData.steps;

    // Initialize the progress bar
    initProgress(totalSteps);

    // POST, use response to listen for events
    fetch(form.action, {
        method: form.method,
        headers: new Headers({'content-type': 'application/json'}),
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        var dreamId = data.dreamId;
        socket.emit('join_room', { 'room': dreamId });
    });


    // // Post as JSON, using Fetch streaming to get results
    // fetch(form.action, {
    //     method: form.method,
    //     headers: new Headers({'content-type': 'application/json'}),
    //     body: JSON.stringify(formData),
    // }).then(async (response) => {
    //     const reader = response.body.getReader();

    //     // TODO: this request should return the dreamId to track
    //     var dreamId = ''

    //     let noOutputs = true;
    //     while (true) {
    //         let {value, done} = await reader.read();
    //         value = new TextDecoder().decode(value);
    //         if (done) {
    //             progressSectionEle.style.display = 'none';
    //             break;
    //         }

    //         for (let event of value.split('\n').filter(e => e !== '')) {
    //             const data = JSON.parse(event);

    //             if (data.event === 'started') {
    //                 noOutputs = false
    //                 document.getElementById("log-container").append('<span>' + data.dreamId + '</span>');
    //                 socket.emit('join_room', { 'room': data.dreamId });
    //                 dreamId = data.dreamId
    //             } else if (data.event === 'result') {
    //                 noOutputs = false;
    //                 document.querySelector("#no-results-message")?.remove();
    //                 appendOutput(data.url, data.seed, data.config);
    //                 progressEle.setAttribute('value', 0);
    //                 progressEle.setAttribute('max', totalSteps);
    //             } else if (data.event === 'upscaling-started') {
    //                 document.getElementById("processing_cnt").textContent=data.processed_file_cnt;
    //                 document.getElementById("scaling-inprocess-message").style.display = "block";
    //             } else if (data.event === 'upscaling-done') {
    //                 document.getElementById("scaling-inprocess-message").style.display = "none";
    //             } else if (data.event === 'step') {
    //                 progressEle.setAttribute('value', data.step);
    //                 if (data.url) {
    //                     progressImageEle.src = data.url;
    //                 }
    //             } else if (data.event === 'canceled') {
    //                 // avoid alerting as if this were an error case
    //                 noOutputs = false;
    //             } else if (data.event === 'done') {
    //                 noOutputs = false;
    //             }
    //         }
    //     }

    //     // Re-enable form, remove no-results-message
    //     form.querySelector('fieldset').removeAttribute('disabled');
    //     document.querySelector("#prompt").value = prompt;
    //     document.querySelector('progress').setAttribute('value', '0');

    //     if (noOutputs) {
    //         alert("Error occurred while generating.");
    //     }

    //     if (dreamId != '')
    //     {
    //       socket.emit('leave_room', { 'room': dreamId });
    //     }
    // });

    // Disable form while generating
    form.querySelector('fieldset').setAttribute('disabled','');
    //document.querySelector("#prompt").value = `Generating: "${prompt}"`;
}

// Socket listeners
socket.on('job_started', (data) => {})

socket.on('dream_result', (data) => {
  var jobId = data.jobId;
  var dreamId = data.dreamId;
  var dreamRequest = data.dreamRequest;
  var src = 'api/images/' + dreamId;

  appendOutput(src, dreamRequest.seed, dreamRequest);


  // var alt = JSON.stringify(dreamRequest);// 'TODO: FILL THIS OUT ' + dreamId;

  // // Create the result image
  // img = document.createElement('img');
  // img.id = jobId + '-' + dreamId;
  // img.src = src;
  // img.alt = alt;
  // var container = document.getElementById("log-container");
  // container.prepend(img);

  // // Remove progress image (might change if we can do multiple at a time in the future)
  // var img = document.getElementById(jobId + '-progress');
  // if (img) {
  //   img.remove();
  // }

  resetProgress();
})

socket.on('dream_progress', (data) => {
  // TODO: it'd be nice if we could get a seed reported here, but the generator would need to be updated
  var step = data.step;
  var totalSteps = data.totalSteps;
  var jobId = data.jobId;
  var dreamId = data.dreamId;
  var src = data.hasProgressImage ?
    'api/intermediates/' + dreamId + '/' + step
    : null;
  setProgress(step, totalSteps, src);
  
  // var hasProgressImage = data.hasProgressImage
  // if (hasProgressImage) {
  //   var src = 'api/intermediates/' + dreamId + '/' + step;
  //   var alt = 'TODO: FILL THIS OUT ' + dreamId + '/' + step;

  //   // Create or update the progress image
  //   var img_id = jobId + '-progress';
  //   var img = document.getElementById(img_id);
  //   var imgExists = true;
  //   if (!img) {
  //     imgExists = false;
  //     img = document.createElement('img');
  //     img.id = img_id;
  //   }
  
  //   img.src = src;
  //   img.alt = alt;
    
  //   if (!imgExists) {
  //     var container = document.getElementById("log-container");
  //     container.prepend(img);
  //   }
  // }
})

socket.on('job_canceled', (data) => {
  resetForm();
  resetProgress();
})

socket.on('job_done', (data) => {
  jobId = data.jobId
  socket.emit('leave_room', { 'room': jobId });

  resetForm();
  resetProgress();
})

window.onload = () => {
    document.querySelector("#generate-form").addEventListener('submit', (e) => {
        e.preventDefault();
        const form = e.target;

        generateSubmit(form);
    });
    document.querySelector("#generate-form").addEventListener('change', (e) => {
        saveFields(e.target.form);
    });
    document.querySelector("#reset-seed").addEventListener('click', (e) => {
        document.querySelector("#seed").value = 0;
        saveFields(e.target.form);
    });
    document.querySelector("#reset-all").addEventListener('click', (e) => {
        clearFields(e.target.form);
    });
    document.querySelector("#remove-image").addEventListener('click', (e) => {
        initimg.value=null;
    });
    loadFields(document.querySelector("#generate-form"));

    document.querySelector('#cancel-button').addEventListener('click', () => {
        fetch('/cancel').catch(e => {
            console.error(e);
        });
    });

    if (!config.gfpgan_model_exists) {
        document.querySelector("#gfpgan").style.display = 'none';
    }
};
