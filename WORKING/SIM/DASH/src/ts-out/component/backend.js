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
import * as request from 'request';
import * as retryrequest from 'requestretry';
import { logging } from '../common/logger';
var logger = logging('BackendShim');
/**
 * Request wrapper class.
 *
 * Exposes callbacks for each part of the lifecycle of a request:
 *  - for all request types: onSend, onFail, onSuccess(& onSuccessResponse)
 *  - for native requests: onProgress
 *  - for native GETS: afterSend, onAbort
 *
 * Exposes the underlying request's send function.
 *
 * For debugging, the log() function will enable automatic logging in the `BackendShim` log.
 */
var Request = /** @class */ (function () {
    function Request(shim) {
        this._shim = shim;
        this._onBody = function (body) { };
        this._onResponse = function (request) { };
        this._error = function () { };
        this._onSend = function (url, content) { };
        this._afterSend = function (request) { };
        this._onAbort = function (request) { };
        this._progress = function (event) { };
        this._log = false;
        // underlying request object
        this.request = undefined;
    }
    Request.prototype._nativeSyncPost = function (path, resource, content) {
        var _this = this;
        if (this._log) {
            logger.log('Sending native sync POST request', path + resource);
        }
        var xhr = new XMLHttpRequest();
        xhr.open("POST", path + resource, false);
        xhr.onreadystatechange = function () {
            if (xhr.readyState == 4 && xhr.status == 200) {
                _this._onResponse(xhr);
            }
        };
        xhr.send(JSON.stringify(content));
        return this;
    };
    Request.prototype._nativeGet = function (path, resource, responseType) {
        var _this = this;
        if (this._log) {
            logger.log('Sending native GET request', path + resource);
        }
        var xhr = new XMLHttpRequest();
        logger.log('GET', path + resource);
        xhr.open('GET', path + resource);
        if (responseType !== undefined) {
            xhr.responseType = responseType;
        }
        xhr.onload = function () {
            if (xhr.status == 200) {
                _this._onResponse(xhr);
            }
            else {
                _this._error();
            }
        };
        xhr.onerror = function () {
            _this._error();
        };
        xhr.onprogress = function (event) {
            _this._progress(event);
        };
        xhr.onabort = function () {
            _this._onAbort(xhr);
        };
        this._onSend(path + resource, undefined);
        xhr.send();
        this._afterSend(xhr);
        this.request = xhr;
        return this;
    };
    Request.prototype._request = function (requestFunc, path, resource, content) {
        var _this = this;
        if (this._log) {
            logger.log('sending request', path + resource, content);
        }
        this._onSend(path + resource, content);
        if (requestFunc === request.get || requestFunc === request.post) {
            this.request = requestFunc(path + resource, content, function (error, res, body) {
                if (error) {
                    logger.log(error);
                    _this._error();
                    return;
                }
                var statusCode = res.statusCode;
                if (statusCode != 200) {
                    logger.log("status code ".concat(statusCode), res, body);
                    _this._error();
                    return;
                }
                if (_this._log) {
                    logger.log('successful request');
                }
                _this._onBody(body);
                _this._onResponse(res);
            });
        }
        else if (requestFunc === retryrequest.post) {
            this.request = requestFunc(path + resource, {
                'uri': path + resource,
                'method': 'POST',
                'json': content,
                'maxAttempts': 1,
                'retryDelay': 100000,
            }, function (error, res, body) {
                if (error) {
                    logger.log(error);
                    _this._error();
                    return;
                }
                var statusCode = res.statusCode;
                if (statusCode != 200) {
                    logger.log("status code ".concat(statusCode), res, body);
                    _this._error();
                    return;
                }
                if (_this._log) {
                    logger.log('successful request');
                }
                _this._onBody(body);
                _this._onResponse(res);
            });
        }
        return this;
    };
    // Builder pattern methods below
    // -----------------------------
    Request.prototype.onProgress = function (callback) {
        // only works for native requests
        this._progress = callback;
        return this;
    };
    Request.prototype.onSend = function (callback) {
        this._onSend = callback;
        return this;
    };
    Request.prototype.afterSend = function (callback) {
        // only works for native GET requests
        this._afterSend = callback;
        return this;
    };
    Request.prototype.onAbort = function (callback) {
        // only works for native GET requests
        this._onAbort = callback;
        return this;
    };
    Request.prototype.onFail = function (callback) {
        this._error = callback;
        return this;
    };
    Request.prototype.onSuccess = function (callback) {
        this._onBody = callback;
        return this;
    };
    Request.prototype.onSuccessResponse = function (callback) {
        this._onResponse = callback;
        return this;
    };
    Request.prototype.log = function () {
        this._log = true;
        return this;
    };
    Object.defineProperty(Request.prototype, "shim", {
        // Getters below
        // -------------
        get: function () {
            return this._shim;
        },
        enumerable: false,
        configurable: true
    });
    return Request;
}());
export { Request };
/**
 * Metrics POST via the node `request` library.
 */
var MetricsRequest = /** @class */ (function (_super) {
    __extends(MetricsRequest, _super);
    function MetricsRequest(shim) {
        var _this = _super.call(this, shim) || this;
        _this._json = {};
        return _this;
    }
    MetricsRequest.prototype.addStats = function (stats) {
        this._json['stats'] = stats;
        return this;
    };
    MetricsRequest.prototype.send = function () {
        return this._request(request.post, this.shim.path, "", {
            json: this._json,
        });
    };
    return MetricsRequest;
}(Request));
export { MetricsRequest };
/**
 * Experimental setup repeated POST via the node `request` library that marks that the DASH player
 * started running.
 */
var StartLoggingRequest = /** @class */ (function (_super) {
    __extends(StartLoggingRequest, _super);
    function StartLoggingRequest(shim) {
        return _super.call(this, shim) || this;
    }
    StartLoggingRequest.prototype.send = function () {
        return this._request(retryrequest.post, this.shim.experimentPath, "/start", {
            'start': true,
        });
    };
    return StartLoggingRequest;
}(Request));
export { StartLoggingRequest };
/**
 * Experimental setup native synchronous POST that requests a decision.
 *
 * To be used by `algo/remote`.
 */
var FrontEndDecisionRequest = /** @class */ (function (_super) {
    __extends(FrontEndDecisionRequest, _super);
    function FrontEndDecisionRequest(shim) {
        var _this = _super.call(this, shim) || this;
        _this._object = {};
        return _this;
    }
    FrontEndDecisionRequest.prototype.addLastFetchTime = function (time) {
        this._object['last_fetch_time'] = time;
        return this;
    };
    FrontEndDecisionRequest.prototype.addIndex = function (index) {
        this._object['index'] = index;
        return this;
    };
    FrontEndDecisionRequest.prototype.addBuffer = function (buffer) {
        this._object['buffer'] = buffer;
        return this;
    };
    FrontEndDecisionRequest.prototype.addRebuffer = function (rebuffer) {
        this._object['rebuffer'] = rebuffer;
        return this;
    };
    FrontEndDecisionRequest.prototype.addBandwidth = function (bandwidth) {
        this._object['bandwidth'] = bandwidth;
        return this;
    };
    FrontEndDecisionRequest.prototype.send = function () {
        return this._nativeSyncPost(this.shim.experimentPath, "/decision", this._object);
    };
    return FrontEndDecisionRequest;
}(Request));
export { FrontEndDecisionRequest };
/**
 * Experimental setup repeated POST via the node `request` library that marks that can be used
 * to send periodic statistic to the experimental pipeline for metrics computation.
 */
var MetricsLoggingRequest = /** @class */ (function (_super) {
    __extends(MetricsLoggingRequest, _super);
    function MetricsLoggingRequest(shim) {
        return _super.call(this, shim) || this;
    }
    MetricsLoggingRequest.prototype.addLogs = function (logs) {
        this._json['logs'] = logs;
        return this;
    };
    MetricsLoggingRequest.prototype.addComplete = function () {
        this._json['complete'] = true;
        return this;
    };
    MetricsLoggingRequest.prototype.send = function () {
        return this._request(retryrequest.post, this.shim.experimentPath, "/metrics", {
            json: this._json,
        });
    };
    return MetricsLoggingRequest;
}(MetricsRequest));
export { MetricsLoggingRequest };
/**
 * *Backend* GET via the node `request` library asking for a `Decision` for a particular `index`.
 */
var PieceRequest = /** @class */ (function (_super) {
    __extends(PieceRequest, _super);
    function PieceRequest(shim) {
        return _super.call(this, shim) || this;
    }
    PieceRequest.prototype.addIndex = function (index) {
        this.index = index;
        return this;
    };
    PieceRequest.prototype.send = function () {
        if (this.index === undefined) {
            throw new TypeError("PieceRequest made without index: ".concat(this));
        }
        return this._request(request.get, this.shim.path, "/" + this.index, {});
    };
    return PieceRequest;
}(Request));
export { PieceRequest };
/**
 * *Backend* GET via the node `request` library asking to abort the request for an `index`.
 */
var AbortRequest = /** @class */ (function (_super) {
    __extends(AbortRequest, _super);
    function AbortRequest() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    AbortRequest.prototype.send = function () {
        if (this.index === undefined) {
            throw new TypeError("PieceRequest made without index: ".concat(this));
        }
        return this._request(request.get, this.shim.abortPath, "/" + this.index, {});
    };
    return AbortRequest;
}(PieceRequest));
export { AbortRequest };
/**
 * *Backend* native GET asking for the header of a quality track.
 *
 * The backend will return a response in the form of an `arraybuffer`.
 */
var HeaderRequest = /** @class */ (function (_super) {
    __extends(HeaderRequest, _super);
    function HeaderRequest(shim) {
        return _super.call(this, shim) || this;
    }
    HeaderRequest.prototype.addQuality = function (quality) {
        this.quality = quality;
        return this;
    };
    HeaderRequest.prototype.send = function () {
        if (this.quality === undefined) {
            throw new TypeError("HeaderRequest made without quality: ".concat(this));
        }
        var resource = "/video".concat(this.quality, "/init.mp4");
        return this._nativeGet(this.shim.basePath, resource, "arraybuffer");
    };
    return HeaderRequest;
}(Request));
export { HeaderRequest };
/**
 * *Backend* native GET asking for a segment.
 *
 * The backend will return a response in the form of an `arraybuffer`.
 */
var ResourceRequest = /** @class */ (function (_super) {
    __extends(ResourceRequest, _super);
    function ResourceRequest(shim) {
        return _super.call(this, shim) || this;
    }
    ResourceRequest.prototype.addIndex = function (index) {
        this.index = index;
        return this;
    };
    ResourceRequest.prototype.send = function () {
        if (this.index === undefined) {
            throw new TypeError("ResourceRequest made without index: ".concat(this));
        }
        return this._nativeGet(this.shim.resourcePath, "/" + this.index, "arraybuffer");
    };
    return ResourceRequest;
}(Request));
export { ResourceRequest };
/**
 * BackendShim that builds all possible request types enumerated above, that is:
 *  - HeaderRequest
 *  - MetricsRequest
 *  - FrontEndDecisionRequest
 *  - StartLoggingRequest
 *  - MetricsLoggingRequest
 *  - PieceRequest
 *  - ResourceRequest
 *  - AbortRequest
 *
 */
var BackendShim = /** @class */ (function () {
    function BackendShim(site, metrics_port, quic_port) {
        // the base path refers to the backend server
        this._base_path = "https://".concat(site, ":").concat(quic_port);
        this._path = "https://".concat(site, ":").concat(quic_port, "/request");
        this._abort = "https://".concat(site, ":").concat(quic_port, "/abort");
        this._resource_path = "https://".concat(site, ":").concat(quic_port, "/piece");
        // the experiment path is used as a logging and control service
        this._experiment_path = "https://".concat(site, ":").concat(metrics_port);
    }
    BackendShim.prototype.headerRequest = function () {
        return new HeaderRequest(this);
    };
    BackendShim.prototype.metricsRequest = function () {
        return new MetricsRequest(this);
    };
    BackendShim.prototype.frontEndDecisionRequest = function () {
        return new FrontEndDecisionRequest(this);
    };
    BackendShim.prototype.startLoggingRequest = function () {
        return new StartLoggingRequest(this);
    };
    BackendShim.prototype.metricsLoggingRequest = function () {
        return new MetricsLoggingRequest(this);
    };
    BackendShim.prototype.pieceRequest = function () {
        return new PieceRequest(this);
    };
    BackendShim.prototype.resourceRequest = function () {
        return new ResourceRequest(this);
    };
    BackendShim.prototype.abortRequest = function () {
        return new AbortRequest(this);
    };
    Object.defineProperty(BackendShim.prototype, "basePath", {
        get: function () {
            return this._base_path;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(BackendShim.prototype, "resourcePath", {
        get: function () {
            return this._resource_path;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(BackendShim.prototype, "path", {
        get: function () {
            return this._path;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(BackendShim.prototype, "abortPath", {
        get: function () {
            return this._abort;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(BackendShim.prototype, "experimentPath", {
        get: function () {
            return this._experiment_path;
        },
        enumerable: false,
        configurable: true
    });
    return BackendShim;
}());
export { BackendShim };
//# sourceMappingURL=backend.js.map