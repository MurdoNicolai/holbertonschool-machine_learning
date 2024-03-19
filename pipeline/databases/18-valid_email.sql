-- resets the attribute valid_email only when the email has been changed
DELIMITER // -- Define delimiter for trigger creation

CREATE TRIGGER after_order_insert
AFTER INSERT ON orders_items
FOR EACH ROW
BEGIN
  UPDATE items
  SET quantity = quantity - NEW.quantity
  WHERE id = NEW.item_id;
END;
//

DELIMITER ; -- Reset delimiter to semicolon
