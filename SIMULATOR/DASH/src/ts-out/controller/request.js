import { logging } from '../common/logger';
var logger = logging('RequestController');
/**
 * Controller responsible for maintaining a fixed pool of long-polling requests.
 *
 * It contains 2 pools of requests(symetric to each other):
 *   - a pull of piece requests: high level-HTTP requests with JSON content
 *   - a pull of resource requests: XMLHttp native requests
 */
var RequestController = /** @class */ (function () {
    function RequestController(videoInfo, shim, pool) {
        this._current = 0;
        this._pool = pool;
        this._shim = shim;
        this._index = 0;
        this._resourceSend = function (index, url, content) { };
        this._resourceProgress = function (index, event) { };
        this._resourceSuccess = function (index, res) { };
        this._pieceSuccess = function (index, body) { };
        this._pieceRequests = {};
        this._resourceRequests = {};
        this._max_index = videoInfo.info[videoInfo.bitrateArray[0]].length;
    }
    /**
     * Getter for piece requests.
     */
    RequestController.prototype.getPieceRequest = function (index) {
        return this._pieceRequests[index];
    };
    /**
     * Geter for resource requests.
     */
    RequestController.prototype.getResourceRequest = function (index) {
        return this._resourceRequests[index];
    };
    RequestController.prototype._pieceRequest = function (index) {
        var _this = this;
        this._pieceRequests[index] = this._shim
            .pieceRequest()
            .addIndex(index)
            .onSuccess(function (body) {
            _this._pieceSuccess(index, body);
        }).onFail(function () {
            throw new Error("Piece request ".concat(index, " failed"));
        }).send();
    };
    RequestController.prototype._resourceRequest = function (index) {
        var _this = this;
        this._resourceRequests[index] = this._shim
            .resourceRequest();
        this._resourceRequests[index]
            .addIndex(index)
            .onSend(function (url, content) {
            _this._resourceSend(index, url, content);
        })
            .afterSend(function (request) {
            _this._resourceAfterSend(index, request);
        })
            .onProgress(function (event) {
            _this._resourceProgress(index, event);
        })
            .onAbort(function (request) {
            _this._resourceOnAbort(index, request);
        })
            .onSuccessResponse(function (res) {
            _this._resourceSuccess(index, res);
            _this._current -= 1;
            _this._request();
        }).onFail(function () {
            throw new Error("Resource request ".concat(index, " failed"));
        }).send();
    };
    RequestController.prototype._request = function () {
        logger.log("indexes", this._index, this._max_index);
        if (this._current < this._pool && this._index < this._max_index) {
            this._current += 1;
            this._index += 1;
        }
        else {
            return;
        }
        var index = this._index;
        this._pieceRequest(index);
        this._resourceRequest(index);
        this._request();
    };
    /**
     * Starts the asynchrnous pool of requests. As both piece and resource requests finish for a piece
     * they get replaces with requests for the pieces with the next index.
     */
    RequestController.prototype.start = function () {
        this._request();
        return this;
    };
    /**
     * Allows attaching a *single* callback before sending a resource request.
     */
    RequestController.prototype.onResourceSend = function (callback) {
        this._resourceSend = callback;
        return this;
    };
    /**
     * Allows attaching a *single* callback after sending a resource request.
     */
    RequestController.prototype.afterResourceSend = function (callback) {
        this._resourceAfterSend = callback;
        return this;
    };
    /**
     * Allows attaching a *single* callback after a resource request was aborted.
     */
    RequestController.prototype.onResourceAbort = function (callback) {
        this._resourceOnAbort = callback;
        return this;
    };
    /**
     * Allows attaching a *single* callback after the browser dispache an update event
     * on the XMLHttp request associated with a resource request.
     */
    RequestController.prototype.onResourceProgress = function (callback) {
        this._resourceProgress = callback;
        return this;
    };
    /**
     * Allows attaching a *single* callback after the resource request has successfully
     * finished.
     */
    RequestController.prototype.onResourceSuccess = function (callback) {
        this._resourceSuccess = callback;
        return this;
    };
    /**
     * Allows attaching a *single* callback before a piece request was made.
     */
    RequestController.prototype.onPieceSuccess = function (callback) {
        this._pieceSuccess = callback;
        return this;
    };
    return RequestController;
}());
export { RequestController };
//# sourceMappingURL=request.js.map