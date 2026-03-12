import { logging } from '../common/logger';
var logger = logging('PieceCache');
/**
 * A dictionary-based cache for Pieces.
 */
var PieceCache = /** @class */ (function () {
    function PieceCache() {
        this.container = {};
    }
    /**
     * Retrieve a piece by the segment index.
     */
    PieceCache.prototype.piece = function (index) {
        return this.container[index];
    };
    /**
     * Insert a piece in the cache. If a piece is already present replace it
     * based on the supplied timestamp.
     */
    PieceCache.prototype.insert = function (piece) {
        if (this.container[piece.index]) {
            var currentPiece = this.container[piece.index];
            if (currentPiece.timestamp < piece.timestamp) {
                this.container[piece.index] = piece;
            }
        }
        else {
            this.container[piece.index] = piece;
        }
    };
    return PieceCache;
}());
export { PieceCache };
//# sourceMappingURL=cache.js.map