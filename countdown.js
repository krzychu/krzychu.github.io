"use strict";
function currentPeriod(type, now) {
    const currentMonth = now.getMonth();
    const year = now.getFullYear();
    let start;
    let end;
    if (type === 'quarter') {
        const startMonth = Math.floor(currentMonth / 3) * 3;
        start = new Date(year, startMonth, 1);
        end = new Date(year, startMonth + 3, 0, 23, 59, 59, 999);
    }
    else if (type === 'semester') {
        if (currentMonth >= 9) {
            start = new Date(year, 9, 1);
            end = new Date(year + 1, 3, 0, 23, 59, 59, 999);
        }
        else if (currentMonth <= 2) {
            start = new Date(year - 1, 9, 1);
            end = new Date(year, 3, 0, 23, 59, 59, 999);
        }
        else {
            start = new Date(year, 3, 1);
            end = new Date(year, 9, 0, 23, 59, 59, 999);
        }
    }
    else if (type === 'half-year') {
        if (currentMonth <= 5) {
            start = new Date(year, 0, 1);
            end = new Date(year, 6, 0, 23, 59, 59, 999);
        }
        else {
            start = new Date(year, 6, 1);
            end = new Date(year + 1, 0, 0, 23, 59, 59, 999);
        }
    }
    else if (type === 'year') {
        start = new Date(year, 0, 1);
        end = new Date(year + 1, 0, 0, 23, 59, 59, 999);
    }
    else {
        throw new Error(`Invalid period type: ${type}`);
    }
    const totalMs = end.getTime() - start.getTime();
    const elapsedMs = now.getTime() - start.getTime();
    const remainingMs = end.getTime() - now.getTime();
    const weeksTillEnd = remainingMs / (1000 * 60 * 60 * 24 * 7);
    const fractionRemaining = remainingMs / totalMs;
    return {
        start: start,
        end: end,
        weeksTillEnd: weeksTillEnd,
        fractionRemaining: fractionRemaining,
    };
}
function getElementByClassName(parent, className) {
    const elements = parent.getElementsByClassName(className);
    if (elements.length != 1) {
        throw new Error(`${className} not found`);
    }
    return elements[0];
}
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}
function displayPeriodOn(element, period) {
    getElementByClassName(element, "start").innerText = `started on ${formatDate(period.start)}`;
    getElementByClassName(element, "end").innerText = `ends on ${formatDate(period.end)}`;
    getElementByClassName(element, "weeks").innerText = `ends in ${period.weeksTillEnd.toFixed(2)} weeks`;
    getElementByClassName(element, "fraction").innerText = `${(period.fractionRemaining * 100).toFixed(2)}% remaining`;
}
function main() {
    const now = new Date();
    const types = ["quarter", "semester", "half-year", "year"];
    for (const type of types) {
        const element = document.getElementById(type);
        displayPeriodOn(element, currentPeriod(type, now));
    }
}
document.addEventListener("DOMContentLoaded", main);
