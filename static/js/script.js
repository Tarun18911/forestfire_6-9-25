// Restrict browser navigation (Alt+Left allowed, Alt+Right blocked)
document.addEventListener('DOMContentLoaded', function() {
    // Disable forward navigation
    history.pushState(null, null, document.URL);
    window.addEventListener('popstate', function() {
        history.pushState(null, null, document.URL);
        // Allow back navigation but not forward
        if (history.length > 1) {
            history.back();
        }
    });
    
    // Prevent specific keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Disable Alt+Right arrow (forward navigation)
        if (e.altKey && e.key === "ArrowRight") {
            e.preventDefault();
            return false;
        }
        
        // Also check for keyCode for older browsers
        if (e.altKey && e.keyCode === 39) {
            e.preventDefault();
            return false;
        }
    });
});