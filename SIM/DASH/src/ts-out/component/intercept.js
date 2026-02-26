var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
import { logging } from '../common/logger';
var logger = logging('Intercept');
export function makeHeader(quality) {
    return "HEADER".concat(quality);
}
/**
 * UrlProcessor class that allows extration of `quality` and `index` from a default  XmlHTTP
 * request made by DASH.
 */
var UrlProcessor = /** @class */ (function () {
    function UrlProcessor(_max_rates, _url) {
        var e_1, _a;
        this.max_rates = _max_rates;
        var url = "".concat(_url);
        try {
            for (var _b = __values(['http://', 'https://']), _c = _b.next(); !_c.done; _c = _b.next()) {
                var prefix = _c.value;
                if (url.includes(prefix)) {
                    url = url.split(prefix)[1];
                }
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
            }
            finally { if (e_1) throw e_1.error; }
        }
        this.url = url;
    }
    Object.defineProperty(UrlProcessor.prototype, "quality", {
        get: function () {
            try {
                return parseInt(this.url.split('/')[1].split('video')[1]);
            }
            catch (err) {
                return undefined;
            }
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(UrlProcessor.prototype, "index", {
        get: function () {
            try {
                return parseInt(this.url.split('/')[2].split('.')[0]);
            }
            catch (err) {
                return undefined;
            }
        },
        enumerable: false,
        configurable: true
    });
    return UrlProcessor;
}());
/**
 * Intercetor utility that can be used for modification of a XMLHttp object.
 */
var InterceptorUtil = /** @class */ (function () {
    function InterceptorUtil() {
    }
    /**
     * Make an object propety writable or unwritable.
     */
    InterceptorUtil.prototype.makeWritable = function (object, property, writable) {
        var descriptor = Object.getOwnPropertyDescriptor(object, property) || {};
        descriptor.writable = writable;
        Object.defineProperty(object, property, descriptor);
    };
    /**
     * For a given event, execute a callback on that event. To not crash the player,
     * catch and log any errors in event execution.
     */
    InterceptorUtil.prototype.executeCallback = function (callback, event) {
        try {
            if (callback) {
                if (event) {
                    callback(event);
                }
                else {
                    callback(undefined);
                }
            }
        }
        catch (ex) {
            logger.log('Exception in', ex, callback);
        }
    };
    /**
     * For the XMLHttpRequest prototype, create a ProgressEvent of type `type` attached to the
     * XMLHttpRequest with the modified attributes according to dictionary `dict`.
     *
     * Make the event undistiguishable from a native event for DASH.js my modifying the ProgressEvent's
     * default generated properties(currentTarget, srcElement, target and trusted). When the progress
     * event is delivered(i.e. XMLHttpRequest's onprogress called), DASH will see our custom event.
     */
    InterceptorUtil.prototype.newEvent = function (ctx, type, dict) {
        var event = new ProgressEvent(type, dict);
        this.makeWritable(event, 'currentTarget', true);
        this.makeWritable(event, 'srcElement', true);
        this.makeWritable(event, 'target', true);
        this.makeWritable(event, 'trusted', true);
        // @ts-ignore: read-only propoerty
        event.currentTarget = ctx;
        // @ts-ignore: read-only propoerty
        event.srcElement = ctx;
        // @ts-ignore: read-only propoerty
        event.target = ctx;
        // @ts-ignore: read-only propoerty
        event.trusted = true;
        return event;
    };
    return InterceptorUtil;
}());
export { InterceptorUtil };
/**
 * Intercetor class that stops outgoing segment XMLHttpRequests made by DASH(matched by the URL structure)
 * and simulates progress and delivery events after the actual requests to the backend made via the
 * BackendShim have finished.
 *
 * The class allows attaching various callbacks so that our DASH wrapper can interact directly with the
 * backend.
 */
var Interceptor = /** @class */ (function (_super) {
    __extends(Interceptor, _super);
    function Interceptor(videoInfo) {
        var _this = _super.call(this) || this;
        _this._videoInfo = videoInfo;
        _this._onRequest = function (ctx, index) { };
        // map of contexts for onIntercept
        _this._toIntercept = {};
        // map of callbacks for onIntercept
        _this._onIntercept = {};
        // map of exposed context inside intercetor request
        _this._objects = {};
        // map of done requests
        // the retries will not be requested, but only logged
        _this._done = {};
        // set of bypass requests
        _this._bypass = new Set();
        return _this;
    }
    Object.defineProperty(Interceptor.prototype, "videoLength", {
        get: function () {
            return this._videoInfo.info[this._videoInfo.bitrateArray[0]].length;
        },
        enumerable: false,
        configurable: true
    });
    /**
     * Allows for a *single* callback to be made for a particular `index` before the XMLHttpRequest
     * is intercepted. The callback exposes our XMLHttpRequest prototype.
     */
    Interceptor.prototype.onRequest = function (callback) {
        this._onRequest = callback;
        return this;
    };
    /**
     * Allow for a *single* callback to be made for a particular `index` as the send function
     * is called for the intercepted request.
     */
    Interceptor.prototype.onIntercept = function (index, callback) {
        logger.log('Cache updated', index);
        this._onIntercept[index] = callback;
        // if we already have a context on toIntercept we can call
        // the function
        if (this._toIntercept[index] !== null && this._toIntercept[index] !== undefined) {
            var ctx = this._toIntercept[index].ctx;
            callback(this._toIntercept[index]);
        }
        return this;
    };
    /**
     * Trigger a progress event for segment `index` with `loaded` bytes out of `total` bytes.
     */
    Interceptor.prototype.progress = function (index, loaded, total) {
        if (this._toIntercept[index] !== null) {
            var object_1 = this._toIntercept[index];
            var ctx_1 = object_1["ctx"];
            var makeWritable = object_1["makeWritable"];
            var execute = object_1["execute"];
            var newEvent = function (type, dict) {
                return object_1["newEvent"](ctx_1, type, dict);
            };
            // dispach the progress events towards the original request
            makeWritable(ctx_1, 'readyState', true);
            ctx_1.readyState = 3;
            execute(ctx_1.onprogress, newEvent('progress', {
                'lengthComputable': true,
                'loaded': loaded,
                'total': total,
            }));
        }
    };
    /**
     * Return the *context* for a request made at segment `index`:
     *   The context contains:
     *    - ctx: the newXHROpen new prototype
     *    - url: the intercepted request url
     *    - makeWritable, execute, newEvent: attached InterceptorUtil functions
     */
    Interceptor.prototype.context = function (index) {
        return this._objects[index];
    };
    /**
     * Set context for an `index`. Should be genrally called only by the intercetor itself.
     */
    Interceptor.prototype.setContext = function (index, obj) {
        this._objects[index] = obj;
        return this;
    };
    /**
     * Set a bypass over the intercetor filter for single *index*. The bypass works only once.
     *
     * This function will be used for request aborts.
     */
    Interceptor.prototype.setBypass = function (index) {
        this._bypass.add(index);
        return this;
    };
    /**
     * Mark that the onIntercept function has been attached for an `index` and hence needs to be
     * called later by the interceptor.
     */
    Interceptor.prototype.intercept = function (index) {
        this._toIntercept[index] = null;
        return this;
    };
    /**
     * Start the interceptor. After the start function has been called, all the native outgoing XMLHttp
     * requests constructed after this point will be put through the intercetor filter.
     */
    Interceptor.prototype.start = function () {
        var interceptor = this;
        var oldOpen = window.XMLHttpRequest.prototype.open;
        var max_rates = interceptor._videoInfo.bitrates.length;
        // override the open method
        function newXHROpen(method, url, async, user, password) {
            var ctx = this;
            var oldSend = this.send;
            var bypassDetected = false;
            // modify url
            if (url.includes('video') && url.endsWith('.m4s') && !url.includes('Header')) {
                logger.log('To modify', url);
                var processor = new UrlProcessor(max_rates, url);
                var maybeIndex = processor.index;
                if (maybeIndex === undefined) {
                    logger.log("[error] Index not present in ".concat(url));
                }
                else {
                    var index = maybeIndex;
                    if (interceptor._done[index] === undefined) {
                        interceptor._onRequest(ctx, index);
                        interceptor._done[index] = url;
                    }
                    else {
                        if (interceptor._bypass.has(index)) {
                            // Bypass detected
                            logger.log("Bypass detected", index, url);
                            bypassDetected = true;
                            interceptor._onRequest(ctx, index);
                            interceptor._done[index] = url;
                        }
                        else {
                            logger.log("Retry on request", index, url);
                        }
                    }
                }
            }
            if (bypassDetected) {
                return oldOpen.apply(this, arguments);
            }
            ctx.send = function () {
                if (url.includes('video') && url.endsWith('.m4s')) {
                    logger.log(url);
                    var processor = new UrlProcessor(max_rates, url);
                    var maybeIndex = processor.index;
                    var quality = processor.quality;
                    if (maybeIndex === undefined) {
                        logger.log("[error] Index not present in ".concat(url));
                    }
                    else {
                        var index = maybeIndex;
                        if (interceptor._toIntercept[index] !== undefined) {
                            logger.log("intercepted", url, ctx);
                            // adding the context
                            interceptor._toIntercept[index] = {
                                'ctx': ctx,
                                'url': url,
                                'makeWritable': interceptor.makeWritable,
                                'execute': interceptor.executeCallback,
                                'newEvent': interceptor.newEvent,
                            };
                            // if the callback was set this means we already got the new response
                            if (interceptor._onIntercept[index] !== undefined) {
                                interceptor._onIntercept[index](interceptor._toIntercept[index]);
                                return;
                            }
                            return;
                        }
                        else {
                            return oldSend.apply(this, arguments);
                        }
                    }
                }
                else {
                    return oldSend.apply(this, arguments);
                }
            };
            return oldOpen.apply(this, arguments);
        }
        // @ts-ignore: overriding XMLHttpRequest
        window.XMLHttpRequest.prototype.open = newXHROpen;
    };
    return Interceptor;
}(InterceptorUtil));
export { Interceptor };
//# sourceMappingURL=intercept.js.map