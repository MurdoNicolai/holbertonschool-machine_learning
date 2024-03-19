-- creates a trigger that decreases the quantity of an item after adding a new order
DELIMITER // -- Define delimiter for trigger creation

CREATE TRIGGER after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
  UPDATE items
  SET quantity = quantity - NEW.number
  WHERE name = NEW.item_name;
END;
//

DELIMITER ; -- Reset delimiter to semicolon
