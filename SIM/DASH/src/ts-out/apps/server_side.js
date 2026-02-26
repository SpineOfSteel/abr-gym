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
import { Decision, Segment, SEGMENT_STATE } from '../common/data';
import { logging, exportLogs } from '../common/logger';
import { SetQualityController, onEvent } from '../component/abr';
import { Metrics, StatsTracker } from '../component/stats';
import { checking } from '../component/consistency';
import { Interceptor, makeHeader } from '../component/intercept';
import { RequestController } from '../controller/request';
import { QualityController } from '../controller/quality';
import { StatsController } from '../controller/stats';
var logger = logging('App');
var qualityStream = checking('quality');
var metricsStream = checking('metrics');
var POOL_SIZE = 5;
/**
 * Simulate an XMLHttp response for the original DASH request to the backend.
 */
function cacheHit(object, res) {
    var ctx = object.ctx;
    var url = object.url;
    var makeWritable = object.makeWritable;
    var execute = object.execute;
    var newEvent = function (type, dict) {
        return object.newEvent(ctx, type, dict);
    };
    setTimeout(function () {
        // Making response fields writable.
        makeWritable(ctx, 'responseURL', true);
        makeWritable(ctx, 'response', true);
        makeWritable(ctx, 'readyState', true);
        makeWritable(ctx, 'status', true);
        makeWritable(ctx, 'statusText', true);
        // The request emulating the start of the request. This may 
        // have alreay occured by the dispatch of the onprogress events.
        // 
        // We hence check for XMLHttpRequest's readyState(3 means in progress).
        var total = res.response.byteLength;
        if (ctx.readyState !== 3) {
            ctx.readyState = 3;
            execute(ctx.onprogress, newEvent('progress', {
                'lengthComputable': true,
                'loaded': 0,
                'total': total,
            }));
        }
        // Modify the final response to be the arraybuffer that was requested 
        // via the long-polling request.
        ctx.responseType = "arraybuffer";
        ctx.responseURL = url;
        ctx.response = res.response;
        ctx.readyState = 4;
        ctx.status = 200;
        ctx.statusText = "OK";
        logger.log('Overrided', ctx.responseURL);
        // Call the final onprogress event and the onload event that occurs at the 
        // finish of each XMLHttpRequest. This would normally be called natively by the 
        // XMLHttpRequest browser implementations.
        execute(ctx.onprogress, newEvent('progress', {
            'lengthComputable': true,
            'loaded': total,
            'total': total,
        }));
        execute(ctx.onload, newEvent('load', {
            'lengthComputable': true,
            'loaded': total,
            'total': total,
        }));
        execute(ctx.onloadend);
    }, 0);
}
/**
 * Back-End based ABR implementations.
 *
 * Uses an algorithm from the `quic/chromium/src/net/abrcc/abr` folder.
 */
var ServerSideApp = /** @class */ (function () {
    function ServerSideApp(player, recordMetrics, shim, videoInfo) {
        this.shim = shim;
        this.tracker = new StatsTracker(player);
        this.interceptor = new Interceptor(videoInfo);
        this.requestController = new RequestController(videoInfo, this.shim, POOL_SIZE);
        this.qualityController = new QualityController();
        this.statsController = new StatsController();
        this.recordMetrics = recordMetrics;
        this.max_quality = videoInfo.bitrateArray.length;
        this.max_index = videoInfo.info[videoInfo.bitrateArray[0]].length;
        SetQualityController(this.qualityController);
    }
    ServerSideApp.prototype.start = function () {
        var _this = this;
        var _loop_1 = function (quality) {
            var header = makeHeader(quality);
            this_1.shim
                .headerRequest()
                .addQuality(quality)
                .onSend(function (url, content) {
                _this.interceptor.intercept(header);
            })
                .onSuccessResponse(function (res) {
                _this.interceptor.onIntercept(header, function (object) {
                    cacheHit(object, res);
                });
            })
                .send();
        };
        var this_1 = this;
        // Request all headers at the beginning.
        for (var quality = 1; quality <= this.max_quality; quality++) {
            _loop_1(quality);
        }
        // Callback for each successful piece request. Send the decision to the quality
        // stream for consistency checking and update the QualityController cache.
        var onPieceSuccess = function (index, body) {
            var object = JSON.parse(body);
            var decision = new Decision(object.index, object.quality, object.timestamp);
            _this.qualityController.addPiece(decision);
            qualityStream.push(decision);
        };
        // Setup long polling mechanism. 
        this.requestController
            .onPieceSuccess(function (index, body) {
            // When the Decision is received from the backend, start the callback. 
            onPieceSuccess(index, body);
        })
            .onResourceSend(function (index, url, content) {
            // When the resource request was sent, let the interceptor know.
            _this.interceptor.intercept(index);
        })
            .afterResourceSend(function (index, request) {
            // After the resource was sent(i.e. the send method was called), append 
            // set the interceptor context.
            _this.interceptor.setContext(index, {
                'xmlRequest': request,
                'requestObject': _this.requestController.getResourceRequest(index),
            });
        })
            .onResourceProgress(function (index, event) {
            var loaded = event.loaded;
            var total = event.total;
            if (loaded !== undefined && total !== undefined) {
                // As the download progresses, dispatch the segment progress to the intercetor.
                // This will maintain correct metrics for DASH as the `onprogress` events are called.
                _this.interceptor.progress(index, loaded, total);
                // The progress will be registered with the StatsController as well. 
                var segment = new Segment()
                    .withState(SEGMENT_STATE.PROGRESS)
                    .withLoaded(loaded)
                    .withTotal(total)
                    .withIndex(index);
                var metrics = new Metrics()
                    .withSegment(segment);
                _this.statsController.addMetrics(metrics);
            }
        })
            .onResourceSuccess(function (index, res) {
            var quality = _this.qualityController.getQuality(index);
            if (quality === undefined) {
                throw new TypeError("onResourceSuccess - missing quality:"
                    + "index ".concat(index, ", res ").concat(res));
            }
            // When the resource(segment) was full downloaded, we firstly register 
            // the metrics.
            var segment = new Segment()
                .withQuality(quality)
                .withState(SEGMENT_STATE.DOWNLOADED)
                .withIndex(index);
            var metrics = new Metrics()
                .withSegment(segment);
            _this.statsController.addMetrics(metrics);
            // Then, we can dispatch the event to the interceptor. This will cause the original
            // XMLHttp request to finish and dispatch the correct DASH callbacks for updating the 
            // buffer and adjusting the video playback.
            _this.interceptor.onIntercept(index, function (object) {
                cacheHit(object, res);
            });
        })
            .onResourceAbort(function (index, req) {
            // On a resource abort, we need to update the quality stream and the metrics being
            // sent to the experimental setup.
            // WARN: note this behavior is dependent on the ABR rule specific implementation
            //       from the `src/component/abr.js` file.
            // -- in case that changes, we need to invalidate the index for the streams,
            //    rather than modifying the value to 0
            logger.log('Fixing quality stream at index ', index);
            qualityStream.replace(index, 0);
            // Add metrics so that the experimental setup knows about the
            // newly uploaded segment.
            var segment = new Segment()
                .withQuality(0)
                .withState(SEGMENT_STATE.LOADING)
                .withIndex(index);
            var metrics = new Metrics()
                .withSegment(segment);
            logger.log('Sending extra metrics', metrics.serialize(true));
            _this.shim
                .metricsLoggingRequest()
                .addStats(metrics.serialize(true))
                .send();
        })
            .start();
        this.interceptor
            .onRequest(function (ctx, index) {
            if (index == _this.max_index) {
                // Finish stream if we downloaded everything
                eos(ctx);
                return;
            }
            // Only when a request is sent, this means that the next 
            // decision of the ABR component will ask for the next 
            // index.
            _this.qualityController.advance(index + 1);
            // Handle the DASH EventBus that denotes the fragment loading completion.
            // As a frament gets loaded, the ABR rules are prompted for the next segment.
            //
            // In this case, we want to wait for an event that is going to schedule a new piece.
            // This needs to be done if we do not own the decision for the next piece.
            //
            // WARN: Note the details of the implementation below directly interact with DASH 
            // internals specific to version 3.0.0.
            var first = false;
            onEvent("OnFragmentLoadingCompleted", function (context) {
                if (!first) {
                    first = true;
                    // Stop the player until we receive the decision for piece (index + 1)
                    var controller_1 = context.scheduleController;
                    var request_1 = _this.requestController.getPieceRequest(index + 1);
                    if (request_1) {
                        // The request is undefined after we pass all the pieces.
                        logger.log("Scheduling", index + 1);
                        var ended = false;
                        if (request_1.request !== undefined) {
                            ended = request_1.request._ended;
                        }
                        // If the request did not end.
                        if (!ended) {
                            // Keep pooling the controller until the request has ended
                            // and when it did restart the controller.
                            var startAfterEnded_1 = function () {
                                var ended = false;
                                if (request_1.request !== undefined) {
                                    ended = request_1.request._ended;
                                }
                                if (!ended) {
                                    setTimeout(startAfterEnded_1, 10);
                                }
                                else {
                                    controller_1.start();
                                    logger.log("SchedulingController started.");
                                    if (request_1.request === undefined) {
                                        throw new TypeError("missing request for ".concat(request_1));
                                    }
                                    else {
                                        onPieceSuccess(index + 1, request_1.request.response.body);
                                    }
                                }
                            };
                            logger.log("SchedulingController stopped.");
                            controller_1.stop();
                            startAfterEnded_1();
                        }
                        else {
                            if (request_1.request === undefined) {
                                throw new TypeError("missing request for ".concat(request_1));
                            }
                            else {
                                onPieceSuccess(index + 1, request_1.request.response.body);
                            }
                        }
                    }
                }
            });
            // Send metrics to tracker after each new request was sent.
            _this.tracker.getMetrics();
        })
            .start();
        // Listen for stream finishing.
        var eos = function (_unsued) {
            logger.log('End of stream');
            if (_this.recordMetrics) {
                var logs = exportLogs();
                _this.shim
                    .metricsLoggingRequest()
                    .addLogs(logs)
                    .addComplete()
                    .send();
            }
        };
        onEvent("endOfStream", function (context) { return eos(context); });
        onEvent("PLAYBACK_ENDED", function (context) { return eos(context); });
        onEvent("Detected unintended removal", function (context) {
            logger.log('Detected unintdended removal!');
            var controller = context.scheduleController;
            controller.start();
        });
        this.tracker.registerCallback(function (metrics) {
            var e_1, _a;
            // Log metrics
            _this.statsController.addMetrics(metrics);
            // Push segments to the metrics stream for consistency checks
            var allMetrics = _this.statsController.metrics;
            try {
                for (var _b = __values(metrics.segments), _c = _b.next(); !_c.done; _c = _b.next()) {
                    var segment = _c.value;
                    if (segment.state != SEGMENT_STATE.PROGRESS) {
                        metricsStream.push(segment);
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
            if (_this.recordMetrics) {
                // Send metrics without progress segments to the experimental 
                // setup for centralization purposes.
                _this.shim
                    .metricsLoggingRequest()
                    .addStats(allMetrics.serialize(true))
                    .send();
            }
            // Send metrics to the backend.
            _this.shim
                .metricsRequest()
                .addStats(allMetrics.serialize())
                .onSuccess(function (body) {
            }).onFail(function () {
            }).send();
            // Advance the metrics timestamp. This ensures that we will only send fresh information
            // all the time and we will not clutter the pipeline.
            var timestamp = (allMetrics.playerTime.slice(-1)[0] || { 'timestamp': 0 }).timestamp;
            allMetrics.segments.forEach(function (segment) {
                timestamp = Math.max(segment.timestamp, timestamp);
            });
            _this.statsController.advance(timestamp);
        });
        this.tracker.start();
    };
    return ServerSideApp;
}());
export { ServerSideApp };
//# sourceMappingURL=server_side.js.map