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
import { Piece } from '../common/data';
import { logging } from '../common/logger';
var logger = logging('PieceConsistencyChecker');
var STATS_INTERVAL = 20000;
/**
 * Stream that allows multiple callbacks after each push.
 */
var Stream = /** @class */ (function () {
    function Stream() {
        this._callbacks = [];
    }
    Stream.prototype.push = function (value) {
        var e_1, _a;
        try {
            for (var _b = __values(this._callbacks), _c = _b.next(); !_c.done; _c = _b.next()) {
                var callback = _c.value;
                callback(value);
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
            }
            finally { if (e_1) throw e_1.error; }
        }
    };
    Stream.prototype.onPush = function (callback) {
        this._callbacks.push(callback);
    };
    return Stream;
}());
/**
 * Automatic consistency check crass that accepts pushes to any number of streams and
 * checks that the stream values are consistent. The stream classes allow replacements as we allow
 * for segment downloads to be canceled(via Aborts).
 */
var ConsistencyChecker = /** @class */ (function () {
    function ConsistencyChecker() {
        var _this = this;
        this._streams = {};
        this._values = {};
        setInterval(function () {
            _this.__stats();
        }, STATS_INTERVAL);
    }
    ConsistencyChecker.prototype.__stats = function () {
        logger.log("Periodic stats...");
        for (var name_1 in this._streams) {
            var size = Object.keys(this._values[name_1] || {}).length;
            logger.log("stream ".concat(name_1), "size ".concat(size), this._values[name_1]);
        }
    };
    ConsistencyChecker.prototype.__addStream = function (name, stream) {
        var _this = this;
        this._streams[name] = stream;
        this._values[name] = {};
        stream.onPush(function (piece) {
            if (!(piece instanceof Piece)) {
                throw new TypeError("[ConsistencyChecker] wrong type inserted");
            }
            _this._values[name][piece.index] = piece.quality;
            for (var name2 in _this._streams) {
                var q1 = _this._values[name][piece.index];
                var q2 = _this._values[name2][piece.index];
                if (q2 !== undefined && q1 !== q2) {
                    logger.log("Inconsistent qualities for index ".concat(piece.index), "".concat(name, ": ").concat(q1), "".concat(name2, ": ").concat(q2));
                }
            }
        });
    };
    /**
     * Push `value` to stream `name`.
     *
     * After each push, the consistency will be automatically checked with the rest of streams
     * and inconsistencies will be logged.
     */
    ConsistencyChecker.prototype.push = function (name, value) {
        if (this._streams[name] === undefined) {
            this.__addStream(name, new Stream());
        }
        this._streams[name].push(value);
    };
    /**
     * Replace the `index` `value` from stream `name`.
     */
    ConsistencyChecker.prototype.replace = function (name, index, value) {
        this._values[name][index] = value;
    };
    return ConsistencyChecker;
}());
var checker = new ConsistencyChecker();
/**
 * Checker targeted on stream `name`.
 */
var TargetedChecker = /** @class */ (function () {
    function TargetedChecker(name) {
        this.name = name;
    }
    /**
     * Push a value to the stream.
     *
     * After each push, the consistency will be automatically checked with the rest of streams
     * and inconsistencies will be logged.
     */
    TargetedChecker.prototype.push = function (value) {
        checker.push(this.name, value);
    };
    /**
     * Replace a value from the steam.
     */
    TargetedChecker.prototype.replace = function (index, value) {
        checker.replace(this.name, index, value);
    };
    return TargetedChecker;
}());
/**
 * Returns a targetted checker.
 */
export function checking(name) {
    return new TargetedChecker(name);
}
//# sourceMappingURL=consistency.js.map