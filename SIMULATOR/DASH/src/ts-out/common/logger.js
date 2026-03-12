var __values = (this && this.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { timestamp as create_timestamp } from '../common/time';
var loggers = {};
var central_log = new /** @class */ (function () {
    function class_1() {
        this._log = [];
    }
    class_1.prototype.log = function (logname, args) {
        var e_1, _a;
        var timestamp = create_timestamp(new Date());
        var toLog = "[".concat(logname, "] ").concat(timestamp);
        try {
            for (var args_1 = __values(args), args_1_1 = args_1.next(); !args_1_1.done; args_1_1 = args_1.next()) {
                var arg = args_1_1.value;
                toLog = toLog.concat(" | ");
                if (typeof arg === 'string') {
                    toLog = toLog.concat(arg);
                }
                else {
                    toLog = toLog.concat(JSON.stringify(arg));
                }
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (args_1_1 && !args_1_1.done && (_a = args_1.return)) _a.call(args_1);
            }
            finally { if (e_1) throw e_1.error; }
        }
        this._log.push(toLog);
    };
    class_1.prototype.getLogs = function () {
        return __spreadArray([], __read(this._log), false);
    };
    return class_1;
}())();
export function exportLogs() {
    return central_log.getLogs();
}
function hashCode(str) {
    var hash = 0;
    for (var i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return hash;
}
function intToRGB(i) {
    var c = (i & 0x00FFFFFF).toString(16).toUpperCase();
    return "00000".substring(0, 6 - c.length) + c;
}
/**
 * Class that allows logging a variable number of `any` arguments.
 */
var Logger = /** @class */ (function () {
    function Logger(logName) {
        this.logName = logName;
        this.color = intToRGB(hashCode(logName));
    }
    Logger.prototype.log = function () {
        var e_2, _a;
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
        var toLog = [
            "%c  ".concat(this.logName, "  "),
            "color: white; background-color: #".concat(this.color),
        ];
        try {
            for (var args_2 = __values(args), args_2_1 = args_2.next(); !args_2_1.done; args_2_1 = args_2.next()) {
                var argument = args_2_1.value;
                toLog.push(" | ");
                toLog.push(argument);
            }
        }
        catch (e_2_1) { e_2 = { error: e_2_1 }; }
        finally {
            try {
                if (args_2_1 && !args_2_1.done && (_a = args_2.return)) _a.call(args_2);
            }
            finally { if (e_2) throw e_2.error; }
        }
        console.log.apply(console, __spreadArray([], __read(toLog), false));
        central_log.log(this.logName, args);
    };
    return Logger;
}());
export { Logger };
/**
 * Given a project-wide `logName`, create a logger associated with the `logName` to be
 * used in the file in which the logging functionality is invoked.
 *
 * Usage:
 *   const logger = logging('example');
 *      [...]
 *   logger.log('a', 123, {'an' : 'object'});
 */
export function logging(logName) {
    if (loggers[logName] === undefined) {
        loggers[logName] = new Logger(logName);
    }
    return loggers[logName];
}
//# sourceMappingURL=logger.js.map