// Lightbox
(function() {
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    let paths = [];
    let currentIdx = -1;

    function collectPaths() {
        const container = document.querySelector('.overlay:not(.hidden) .overlay-body')
            || document.body;
        return Array.from(container.querySelectorAll('.lightbox-link')).map(a => a.dataset.path);
    }

    function show(path) {
        paths = collectPaths();
        currentIdx = paths.indexOf(path);
        lightboxImg.src = '/image?path=' + encodeURIComponent(path);
        lightbox.classList.remove('hidden');
    }

    function close() {
        lightbox.classList.add('hidden');
        lightboxImg.src = '';
    }

    function navigate(delta) {
        if (!paths.length) return;
        currentIdx = (currentIdx + delta + paths.length) % paths.length;
        lightboxImg.src = '/image?path=' + encodeURIComponent(paths[currentIdx]);
    }

    document.addEventListener('click', function(e) {
        const link = e.target.closest('.lightbox-link');
        if (!link) return;
        e.preventDefault();
        show(link.dataset.path);
    });

    lightbox.querySelector('.lightbox-close').addEventListener('click', close);
    lightbox.querySelector('.overlay-backdrop').addEventListener('click', close);
    lightbox.querySelector('.lightbox-prev').addEventListener('click', function() { navigate(-1); });
    lightbox.querySelector('.lightbox-next').addEventListener('click', function() { navigate(1); });

    document.addEventListener('keydown', function(e) {
        if (lightbox.classList.contains('hidden')) return;
        if (e.key === 'Escape') close();
        else if (e.key === 'ArrowLeft') navigate(-1);
        else if (e.key === 'ArrowRight') navigate(1);
    });
})();

// Related overlay
(function() {
    const overlay = document.getElementById('related-overlay');
    const body = document.getElementById('related-results');

    function renderCard(r, extra) {
        const cls = extra.source ? 'result result-source' : 'result';
        const simText = extra.similarity != null ? ` &mdash; similarity: ${extra.similarity}` : '';
        return `
            <div class="${cls}">
                <div class="thumb">
                    <a href="#" class="lightbox-link" data-path="${r.file_path}">
                        <img src="/image?path=${encodeURIComponent(r.file_path)}&thumb=1" loading="lazy" alt="${r.name}">
                    </a>
                </div>
                <div class="info">
                    <div class="date">${r.date}</div>
                    <div class="meta">${simText ? `similarity: ${extra.similarity} &mdash; ` : ''}<a href="/image?path=${encodeURIComponent(r.file_path)}" download class="action-link">download</a> &mdash; <a href="#" class="related-link" data-id="${r.id}">related</a></div>
                    <div class="meta file-meta">${r.name} &mdash; ${r.width}x${r.height}${r.file_size ? ` &mdash; ${r.file_size}` : ''}</div>
                    <pre class="ocr">${r.ocr_text.replace(/</g, '&lt;')}</pre>
                </div>
            </div>`;
    }

    document.addEventListener('click', function(e) {
        const link = e.target.closest('.related-link');
        if (!link) return;
        e.preventDefault();
        const id = link.dataset.id;
        body.innerHTML = '<p class="loading">Loading...</p>';
        overlay.classList.remove('hidden');

        fetch('/related/' + id)
            .then(r => r.json())
            .then(data => {
                if (!data.related.length) {
                    body.innerHTML = '<p class="loading">No related screenshots found.</p>';
                    return;
                }
                let html = '';
                if (data.source) {
                    html += renderCard(data.source, {source: true});
                }
                html += data.related.map(r => renderCard(r, {similarity: r.similarity})).join('');
                body.innerHTML = html;
            });
    });

    overlay.querySelector('.overlay-close').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });
    overlay.querySelector('.overlay-backdrop').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && document.getElementById('lightbox').classList.contains('hidden')) {
            overlay.classList.add('hidden');
        }
    });
})();
