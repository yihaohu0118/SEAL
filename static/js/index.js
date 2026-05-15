function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (!dropdown || !button) return;

    dropdown.classList.toggle('show');
    button.classList.toggle('active');
}

document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && dropdown && button && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

document.addEventListener('keydown', function(event) {
    if (event.key !== 'Escape') return;

    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown && button) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');

    if (!bibtexElement || !button) return;

    const copyText = button.querySelector('.copy-text');
    const markCopied = function() {
        button.classList.add('copied');
        if (copyText) copyText.textContent = 'Copied';

        setTimeout(function() {
            button.classList.remove('copied');
            if (copyText) copyText.textContent = 'Copy';
        }, 2000);
    };

    if (navigator.clipboard) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(markCopied).catch(function() {
            fallbackCopy(bibtexElement.textContent);
            markCopied();
        });
    } else {
        fallbackCopy(bibtexElement.textContent);
        markCopied();
    }
}

function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;

    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});
