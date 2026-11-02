import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

const VIDEO_TYPES = [
    'video/mp4',
    'video/webm',
    'video/x-matroska',
    'video/quicktime',
    'image/gif',
];

function uploadVideo(file, onProgress) {
    const body = new FormData();
    body.append('image', file, file.name);
    return new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open('POST', api.apiURL('/upload/image'));
        request.upload.onprogress = (event) => {
            if (event.lengthComputable) onProgress?.(event.loaded / event.total);
        };
        request.onerror = () => reject(new Error('Video upload failed.'));
        request.onload = () => {
            if (request.status !== 200) {
                reject(new Error(`${request.status} ${request.statusText}`));
                return;
            }
            resolve(JSON.parse(request.responseText));
        };
        request.send(body);
    });
}

function installVideoPicker(nodeType) {
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        originalCreated?.apply(this, arguments);
        const node = this;
        const pathWidget = node.widgets?.find((widget) => widget.name === 'video_path');
        if (!pathWidget || node.widgets.some((widget) => widget.name === 'video_preview')) return;

        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = VIDEO_TYPES.join(',');
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);

        const preview = document.createElement('video');
        preview.controls = true;
        preview.muted = true;
        preview.loop = true;
        preview.style.width = '100%';
        preview.style.display = 'none';
        const previewWidget = node.addDOMWidget('video_preview', 'preview', preview, {
            serialize: false,
            hideOnZoom: true,
            getValue: () => pathWidget.value,
            setValue: () => {},
        });
        previewWidget.computeSize = (width) => (
            preview.style.display === 'none' ? [width, -4] : [width, 180]
        );

        const setPreview = (filename) => {
            if (!filename) {
                preview.removeAttribute('src');
                preview.style.display = 'none';
            } else {
                const params = new URLSearchParams({
                    filename,
                    type: 'input',
                    t: String(Date.now()),
                });
                preview.src = api.apiURL(`/view?${params}`);
                preview.style.display = 'block';
            }
            node.setDirtyCanvas(true, true);
        };

        const chooseWidget = node.addWidget('button', 'choose video file', 'image', () => {
            fileInput.click();
        });
        chooseWidget.options.serialize = false;

        fileInput.addEventListener('change', async () => {
            const file = fileInput.files?.[0];
            if (!file) return;
            try {
                node.progress = 0;
                const result = await uploadVideo(file, (progress) => {
                    node.progress = progress;
                });
                node.progress = undefined;
                pathWidget.value = result.name;
                pathWidget.callback?.(result.name);
                setPreview(result.name);
            } catch (error) {
                node.progress = undefined;
                alert(error.message || error);
            }
        });

        const originalCallback = pathWidget.callback;
        pathWidget.callback = (value) => {
            originalCallback?.(value);
            setPreview(value);
        };
        setPreview(pathWidget.value);

        const originalRemoved = node.onRemoved;
        node.onRemoved = function () {
            fileInput.remove();
            originalRemoved?.apply(this, arguments);
        };
    };
}

app.registerExtension({
    name: 'ComfyUI-StreamNodes.VideoPicker',
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === 'StreamVideoPath') installVideoPicker(nodeType);
    },
});
