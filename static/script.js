document.addEventListener("DOMContentLoaded", function () {
    const textareas = document.querySelectorAll("textarea");

    textareas.forEach(function (textarea) {
        const counter = document.createElement("p");
        counter.className = "small-text";
        textarea.insertAdjacentElement("afterend", counter);

        function updateCounter() {
            counter.textContent = textarea.value.length + " characters";
        }

        textarea.addEventListener("input", updateCounter);
        updateCounter();
    });
});
